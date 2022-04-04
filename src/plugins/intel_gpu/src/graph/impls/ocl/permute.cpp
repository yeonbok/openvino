// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "permute/permute_kernel_selector.h"
#include "permute/permute_kernel_ref.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct permute_impl : typed_primitive_impl_ocl<permute> {
    using parent = typed_primitive_impl_ocl<permute>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<permute_impl>(*this);
    }

    static primitive_impl* create(const permute_node& arg) {
        std::vector<layout> input_layouts;
        for (auto i : arg.get_dependencies()) {
            input_layouts.push_back(i->get_output_layout());
        }
        prim_kernel_params param_info = prim_kernel_params(arg.get_program().get_id(), arg.get_unique_id(), arg.id(),
                                                           arg.get_primitive()->type_string(), input_layouts, arg.get_output_layout(),
                                                           arg.get_program(), arg.get_fused_primitives(),
                                                           arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto permute_params = get_default_params<kernel_selector::permute_params>(param_info);
        auto permute_optional_params =
            get_default_optional_params<kernel_selector::permute_optional_params>(arg.get_program());

        const auto& permute_order = arg.get_primitive()->permute_order;
        permute_params.order = permute_order;
        auto& kernel_selector = kernel_selector::permute_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(permute_params, permute_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto permute = new permute_impl(arg, best_kernels[0]);

        return permute;
    }
};

namespace detail {

attach_permute_impl::attach_permute_impl() {
    implementation_map<permute>::add(impl_types::ocl, permute_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
