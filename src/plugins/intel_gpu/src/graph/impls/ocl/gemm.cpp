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

namespace cldnn {
namespace ocl {

struct gemm_impl : typed_primitive_impl_ocl<gemm> {
    using parent = typed_primitive_impl_ocl<gemm>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gemm_impl>(*this);
    }

public:
    static primitive_impl* create(const gemm_node& arg, const kernel_impl_params& impl_param) {
        auto desc = arg.get_primitive();
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
            auto output_pshape = output_layout.get_partial_shape();
            auto output_rank = output_pshape.rank().get_length();
            for (size_t i = 0; i != input_layouts.size(); ++i) {
                auto input_layout = input_layouts[i];
                auto input_pshape = input_layout.get_partial_shape();
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
                input_layout.set_partial_shape(input_pshape);
                layouts.push_back(input_layout);
            }
            return layouts;
        };
        auto get_gemm_output_layout = [desc](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto layout = output_layout;
            auto output_pshape = output_layout.get_partial_shape();
            auto output_rank = output_pshape.rank().get_length();
            if (output_rank < 4) {
                auto input0_layout = input_layouts[0];
                auto input1_layout = input_layouts[1];
                bool transpose_input0 = desc->transpose_input0;
                bool transpose_input1 = desc->transpose_input1;

                auto M = !transpose_input0 ? input0_layout.spatial(1) : input0_layout.spatial(0);
                auto N = !transpose_input1 ? input1_layout.spatial(0) : input1_layout.spatial(1);

                auto output_shape = input_layouts[0].get_partial_shape().to_shape();
                for (size_t i = 0; i != input_layouts.size(); ++i) {
                    auto input_pshape = input_layouts[i].get_partial_shape();
                    auto input_shape = input_pshape.to_shape();
                    for (int32_t j = 0; j != input_pshape.rank().get_length(); ++j) {
                        output_shape[j] = std::max(output_shape[j], input_shape[j]);
                    }
                }

                auto get_spatial_idx = [](cldnn::format format, size_t spatial_idx) {
                    const size_t idx = (format::is_grouped(format) ? 3 : 2) + (format.spatial_num() - 1 - spatial_idx);
                    return idx;
                };

                output_shape[get_spatial_idx(layout.format, 0)] = N;
                output_shape[get_spatial_idx(layout.format, 1)] = M;
                layout.set_partial_shape(output_shape);
            }
            return layout;
        };
        const auto input_layouts = get_gemm_input_layouts(impl_param.input_layouts, impl_param.output_layout);
        const auto output_layout = get_gemm_output_layout(input_layouts, impl_param.output_layout);

        auto first_fused_input_idx = input_layouts.size();
        const auto fused_descs = impl_param.fused_desc;
        if (fused_descs.size() > 0) {
            first_fused_input_idx = fused_descs[0].dep_start_idx;
        }
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(impl_param, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        gemm_params.inputs.clear();
        for (size_t i = 0; i < std::min(input_layouts.size(), first_fused_input_idx); i++) {
            gemm_params.inputs.push_back(convert_data_tensor(input_layouts[i]));
        }
        gemm_params.outputs[0] = convert_data_tensor(output_layout);

        gemm_params.alpha = desc->alpha;
        gemm_params.beta = desc->beta;
        gemm_params.transpose_input0 = desc->transpose_input0;
        gemm_params.transpose_input1 = desc->transpose_input1;

        bool is_quantized = true;
        for (auto& input : impl_param.input_layouts)
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
