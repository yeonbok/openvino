// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "shape_of/shape_of_kernel_selector.h"
#include "shape_of/shape_of_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct shape_of_impl : typed_primitive_impl_ocl<shape_of> {
    using parent = typed_primitive_impl_ocl<shape_of>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
    }

    static primitive_impl* create(const shape_of_node& arg) {
        std::vector<layout> input_layouts;
        for (auto i : arg.get_dependencies()) {
            input_layouts.push_back(i->get_output_layout());
        }

        prim_kernel_params param_info = prim_kernel_params(arg.get_program().get_id(), arg.get_unique_id(), arg.id(),
                                                           arg.get_primitive()->type_string(), input_layouts, arg.get_output_layout(),
                                                           arg.get_program(), arg.get_fused_primitives(),
                                                           arg.get_fused_activations_funcs(), arg.get_fused_activations_params());


        auto shape_of_params = get_default_params<kernel_selector::shape_of_params>(param_info);
        auto shape_of_optional_params =
            get_default_optional_params<kernel_selector::shape_of_optional_params>(arg.get_program());

        shape_of_params.input_rank = arg.get_dependency(0).get_output_layout().get_rank();
        shape_of_params.input_dims = arg.get_dependency(0).get_output_layout().get_dims();

        auto& kernel_selector = kernel_selector::shape_of_instance();
        auto best_kernels = kernel_selector.GetBestKernels(shape_of_params, shape_of_optional_params);
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto shape_of = new shape_of_impl(arg, best_kernels[0]);

        return shape_of;
    }
};

namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    implementation_map<shape_of>::add(impl_types::ocl, shape_of_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
