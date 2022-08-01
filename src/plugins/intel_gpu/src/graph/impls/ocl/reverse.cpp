// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"
#include "reverse/reverse_kernel_ref.h"
#include "reverse/reverse_kernel_selector.h"
#include "reverse_inst.h"
#include "impls/implementation_map.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct reverse_impl : typed_primitive_impl_ocl<reverse> {
    using parent = typed_primitive_impl_ocl<reverse>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reverse_impl>(*this);
    }

public:
    static ov::PartialShape getConstPartialShape(const ov::Shape constDims) {
        ov::PartialShape constPartialShape;
        switch (constDims.size()) {
        case 6: constPartialShape = ov::PartialShape{TensorValue(constDims[0]), TensorValue(constDims[1]),
                                            TensorValue(constDims[2]), TensorValue(constDims[3]),
                                            TensorValue(constDims[4]), TensorValue(constDims[5])};
            break;
        case 5: constPartialShape = ov::PartialShape{TensorValue(constDims[0]), TensorValue(constDims[1]),
                                            TensorValue(constDims[2]), TensorValue(constDims[3]), TensorValue(constDims[4])};
            break;
        case 4: constPartialShape = ov::PartialShape{TensorValue(constDims[0]), TensorValue(constDims[1]),
                                            TensorValue(constDims[2]), TensorValue(constDims[3])};
            break;
        case 3: constPartialShape = ov::PartialShape{TensorValue(constDims[0]), TensorValue(constDims[1]),
                                            TensorValue(constDims[2]), 1};
            break;
        case 2: constPartialShape = ov::PartialShape{TensorValue(constDims[0]), TensorValue(constDims[1]), 1, 1};
            break;
        case 1: constPartialShape = ov::PartialShape{1, TensorValue(constDims[0]), 1, 1};
            break;
        case 0: constPartialShape = ov::PartialShape{1, 1, 1, 1};
            break;
        default: IE_THROW() << "Invalid constant blob dimensions";
        }
        return constPartialShape;
    }

    static primitive_impl* create(const reverse_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::reverse_params>(impl_param);

        const auto optional_params =
            get_default_optional_params<kernel_selector::reverse_optional_params>(arg.get_program());

        auto constLayout = arg.input(1).get_output_layout();
        auto oldShape = constLayout.get_partial_shape();
        auto newShape = getConstPartialShape(oldShape.to_shape());
        if (oldShape != newShape) {
            constLayout = layout{newShape, constLayout.data_type, constLayout.format};
        }
        params.inputs.push_back(convert_data_tensor(constLayout));
        params.reverseMode = arg.get_primitive()->mode == reverse_mode::index ? kernel_selector::reverse_mode::index
                                                                              : kernel_selector::reverse_mode::mask;

        const auto& kernel_selector = kernel_selector::reverse_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto reverse = new reverse_impl(arg, best_kernels[0]);

        return reverse;
    }
};

namespace detail {

attach_reverse_impl::attach_reverse_impl() {
    implementation_map<reverse>::add(impl_types::ocl,
                                     reverse_impl::create,
                                     {
                                         std::make_tuple(data_types::f16, format::bfyx),
                                         std::make_tuple(data_types::f16, format::bfzyx),
                                         std::make_tuple(data_types::f16, format::bfwzyx),

                                         std::make_tuple(data_types::f32, format::bfyx),
                                         std::make_tuple(data_types::f32, format::bfzyx),
                                         std::make_tuple(data_types::f32, format::bfwzyx),

                                         std::make_tuple(data_types::i32, format::bfyx),
                                         std::make_tuple(data_types::i32, format::bfzyx),
                                         std::make_tuple(data_types::i32, format::bfwzyx),

                                         std::make_tuple(data_types::i64, format::bfyx),
                                         std::make_tuple(data_types::i64, format::bfzyx),
                                         std::make_tuple(data_types::i64, format::bfwzyx),

                                         std::make_tuple(data_types::i8, format::bfyx),
                                         std::make_tuple(data_types::i8, format::bfzyx),
                                         std::make_tuple(data_types::i8, format::bfwzyx),

                                         std::make_tuple(data_types::u8, format::bfyx),
                                         std::make_tuple(data_types::u8, format::bfzyx),
                                         std::make_tuple(data_types::u8, format::bfwzyx)
                                     });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
