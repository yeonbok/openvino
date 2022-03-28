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

namespace {
// This helper function is needed to convert permute order from IE format (bfyx) into cldnn format (bfxy)
inline std::vector<uint16_t> convert_permute_order(const std::vector<uint16_t>& ie_order, size_t rank = 0) {
    std::vector<uint16_t> ie_order_aligned = ie_order;
    // if order size is less than 4 - fill the rest with just copy
    rank = std::max(rank, (size_t)4);
    for (auto o = ie_order_aligned.size(); o < rank; o++)
        ie_order_aligned.push_back((uint16_t)o);

    std::vector<uint16_t> cldnn_order;
    // 1. Switch permute order values for spatial dims
    for (auto const& o : ie_order_aligned) {
        if (o >= 2)
            cldnn_order.push_back(1 + ie_order_aligned.size() - o);
        else
            cldnn_order.push_back(o);
    }

    // 2. Swap spatial positions
    for (int i = 0; i < (cldnn_order.size() - 2) / 2; i++) {
        std::swap(cldnn_order[2 + i], cldnn_order[1 + cldnn_order.size() - (2 + i)]);
    }

    return cldnn_order;
}
} // namespace

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

        auto permute_order = convert_permute_order(arg.get_primitive()->permute_order, arg.get_output_layout().get_rank());
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
