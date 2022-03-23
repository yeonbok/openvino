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
    static primitive_impl* create(const gemm_node& arg) {
        auto desc = arg.get_primitive();
        const auto& param_info = kernel_impl_params(arg.get_program(), desc, arg.get_unique_id(),
                                                    arg.get_input_layouts(), arg.get_output_layout(),
                                                    arg.get_fused_primitives(),
                                                    arg.get_fused_activations_funcs(), arg.get_fused_activations_params());
        auto gemm_params = get_default_params<kernel_selector::gemm_params>(param_info, 1);
        auto gemm_optional_params =
            get_default_optional_params<kernel_selector::gemm_optional_params>(arg.get_program());

        auto gemmSpecificPartialShape =  [](ov::PartialShape& pshape) {
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
        auto output_layout = arg.get_output_layout();
        auto output_pshape = output_layout.size;
        auto output_rank = output_pshape.rank().get_length();
        std::vector<ov::PartialShape> input_shapes;
        for (size_t i = 0; i < arg.inputs_count(); i++) {
            auto input_layout = arg.input(i).get_output_layout();
            auto input_pshape = input_layout.size;
            auto input_rank = input_pshape.rank().get_length();
            if (input_rank != output_rank || input_rank < 4) {
                if (input_rank == 1) {
                    bool transpose = false;
                    if (i == 0) {
                        transpose = arg.get_primitive()->transpose_input0;
                        input_pshape.insert(input_pshape.begin(), 1);
                    } else {
                        transpose = arg.get_primitive()->transpose_input1;
                        input_pshape.insert(input_pshape.end(), 1);
                    }
                    if (transpose) {
                        std::swap(input_pshape[0], input_pshape[1]);
                    }
                }
                if (input_rank < output_rank)
                    input_pshape.insert(input_pshape.begin(), output_rank - input_rank, 1ul);

                gemmSpecificPartialShape(input_pshape);
            }
            input_layout.size = input_pshape;
            input_shapes.push_back(input_pshape);
            if (i == 0)
                gemm_params.inputs[0] = convert_data_tensor(input_layout);
            else
                gemm_params.inputs.push_back(convert_data_tensor(input_layout));
        }
        if (output_rank < 4) {
            ov::op::v0::MatMul op;
            op.set_transpose_a(arg.get_primitive()->transpose_input0);
            op.set_transpose_b(arg.get_primitive()->transpose_input1);
            std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
            shape_infer(&op, input_shapes, output_shapes);
            output_layout.size = output_shapes[0];
            gemm_params.outputs[0] = convert_data_tensor(output_layout);
        }

        gemm_params.alpha = desc->alpha;
        gemm_params.beta = desc->beta;
        gemm_params.transpose_input0 = desc->transpose_input0;
        gemm_params.transpose_input1 = desc->transpose_input1;

        bool is_quantized = true;
        for (auto& input : arg.get_dependencies())
            is_quantized &= data_type_traits::is_quantized(input->get_output_layout().data_type);

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
