// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct gather_tree_impl : typed_primitive_impl_ocl<gather_tree> {
    using parent = typed_primitive_impl_ocl<gather_tree>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_tree_impl>(*this);
    }

    static primitive_impl* create(const gather_tree_node& arg) {
        std::vector<layout> input_layouts;
        for (auto i : arg.get_dependencies()) {
            input_layouts.push_back(i->get_output_layout());
        }
        prim_kernel_params param_info = prim_kernel_params(arg.get_program().get_id(), arg.get_unique_id(), arg.id(),
                                                           arg.get_primitive()->type_string(), input_layouts, arg.get_output_layout(),
                                                           arg.get_program(), arg.get_fused_primitives(),
                                                           arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto b_params = get_default_params<kernel_selector::gather_tree_params>(param_info, 1);
        auto b_optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.get_dependencies().size(); i++) {
            b_params.inputs.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout(), 1));
        }
        auto desc = arg.get_primitive();

        auto& kernel_selector = kernel_selector::gather_tree_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(b_params, b_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
            "Best_kernel.empty()",
            best_kernels.empty(),
            "Cannot find a proper kernel with this arguments");

        return new gather_tree_impl(arg, best_kernels[0]);
    }
};
namespace detail {
attach_gather_tree_impl::attach_gather_tree_impl() {
    implementation_map<gather_tree>::add(impl_types::ocl, gather_tree_impl::create, {
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
