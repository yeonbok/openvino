// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "one_hot/one_hot_kernel_selector.h"
#include "one_hot/one_hot_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

struct one_hot_impl : typed_primitive_impl_ocl<one_hot> {
    using parent = typed_primitive_impl_ocl<one_hot>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<one_hot_impl>(*this);
    }

    static primitive_impl* create(const one_hot_node& arg, std::shared_ptr<kernel_impl_params> impl_param) {
        const auto& prim = arg.get_primitive();
        auto oh_params = get_default_params<kernel_selector::one_hot_params>(*impl_param, 1);
        auto oh_optional_params =
            get_default_optional_params<kernel_selector::one_hot_optional_params>(arg.get_program());

        oh_params.one_hot_axis = prim->one_hot_axis;
        oh_params.on_value = prim->on_value;
        oh_params.off_value = prim->off_value;

        auto output_sizes = impl_param->output_layout.get_dims();

        oh_params.one_hot_limit = output_sizes[oh_params.one_hot_axis];

        auto& kernel_selector = kernel_selector::one_hot_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(oh_params, oh_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");

        return new one_hot_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_one_hot_impl::attach_one_hot_impl() {
    implementation_map<one_hot>::add(impl_types::ocl, one_hot_impl::create, {
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
