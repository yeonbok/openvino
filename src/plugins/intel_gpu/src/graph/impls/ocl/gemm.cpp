// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gemm/gemm_kernel_selector.h"
#include "gemm/gemm_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

#include "matmul_shape_inference.hpp"

namespace cldnn {
namespace ocl {

struct gemm_impl : typed_primitive_impl_ocl<gemm> {
    using parent = typed_primitive_impl_ocl<gemm>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
    }

public:
    static primitive_impl* create(const gemm_node& arg, std::shared_ptr<kernel_impl_params> impl_param) {
        auto desc = arg.get_primitive();
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(*impl_param, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        auto get_gemm_input_layouts = [desc](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto gemm_specific_pshape = [](ov::PartialShape& pshape) {
                switch (pshape.rank().get_length()) {
                    case 2: { // batch, feature representation (rank == 2)
                        pshape.insert(pshape.begin(), 1ul);
                        pshape.insert(pshape.begin(), 1ul);
                        break;
                    }
                    case 3 : { // feature representation (rank == 3)
                        pshape.insert(pshape.begin(), 1, 1ul);
                        break;
                    }
                }
            };
            std::vector<layout> layouts;
            auto output_pshape = output_layout.size;
            auto output_rank = output_pshape.rank().get_length();
            for (size_t i = 0; i != input_layouts.size(); ++i) {
                auto input_layout = input_layouts[i];
                auto input_pshape = input_layout.size;
                auto input_rank = input_pshape.rank().get_length();
                if (input_rank != output_rank || input_rank < 4) {
                    if (input_rank == 1) {
                        bool transpose = false;
                        if (i == 0) {
                            transpose = desc->transpose_input0;
                            input_pshape.insert(input_pshape.begin(), 1);
                        } else {
                            transpose = desc->transpose_input1;
                            input_pshape.insert(input_pshape.end(), 1);
                        }
                        if (transpose) {
                            std::swap(input_pshape[0], input_pshape[1]);
                        }
                    }
                    if (input_rank < output_rank)
                        input_pshape.insert(input_pshape.begin(), output_rank - input_rank, 1ul);

                    gemm_specific_pshape(input_pshape);
                }
                input_layout.size = input_pshape;
                layouts.push_back(input_layout);
            }
            return layouts;
        };

        auto get_gemm_output_layout = [desc](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto layout = output_layout;
            auto output_pshape = output_layout.size;
            auto output_rank = output_pshape.rank().get_length();
            if (output_rank < 4) {
                auto input0_layout = input_layouts[0];
                auto input1_layout = input_layouts[1];
                bool transpose_input0 = desc->transpose_input0;
                bool transpose_input1 = desc->transpose_input1;

                auto M = !transpose_input0 ? input0_layout.spatial(1) : input0_layout.spatial(0);
                auto N = !transpose_input1 ? input1_layout.spatial(0) : input1_layout.spatial(1);

                auto output_shape = input_layouts[0].size.to_shape();
                for (size_t i = 0; i != input_layouts.size(); ++i) {
                    auto input_pshape = input_layouts[i].size;
                    auto input_shape = input_pshape.to_shape();
                    for (size_t j = 0; j != input_pshape.rank().get_length(); ++j) {
                        output_shape[j] = std::max(output_shape[j], input_shape[j]);
                    }
                }
                layout.size = ov::PartialShape(output_shape);
                auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
                    const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
                    return idx;
                };
                layout.size[get_spatial_idx(layout.format, 0)] = N;
                layout.size[get_spatial_idx(layout.format, 1)] = M;
            }
            return layout;
        };
        const auto input_layouts = get_gemm_input_layouts(arg.get_input_layouts(), arg.get_output_layout());
        const auto output_layout = get_gemm_output_layout(input_layouts, arg.get_output_layout());
        const auto& param_info = kernel_impl_params(arg.get_program(), desc, arg.get_unique_id(),
                                                    input_layouts, output_layout,
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(param_info, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            gemm_params.inputs.push_back(convert_data_tensor(param_info.input_layouts[i]));
        }

        gemm_params.alpha = desc->alpha;
        gemm_params.beta = desc->beta;
        gemm_params.transpose_input0 = desc->transpose_input0;
        gemm_params.transpose_input1 = desc->transpose_input1;

        bool is_quantized = true;
        for (auto& input : impl_param->input_layouts)
            is_quantized &= data_type_traits::is_quantized(input.data_type);

        if (is_quantized) {
            gemm_params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            gemm_params.quantization = kernel_selector::QuantizationType::NONE;
        }

        auto& kernel_selector = kernel_selector::gemm_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gemm_params, gemm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new gemm_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_gemm_impl::attach_gemm_impl() {
    implementation_map<gemm>::add(impl_types::ocl, gemm_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
