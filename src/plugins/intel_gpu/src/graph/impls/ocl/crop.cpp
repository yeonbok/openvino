// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct crop_impl : typed_primitive_impl_ocl<crop> {
    using parent = typed_primitive_impl_ocl<crop>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eltwise_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::eltwise_params, kernel_selector::eltwise_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<crop_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<crop>();
        auto params = get_default_params<kernel_selector::eltwise_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::eltwise_optional_params>(impl_param.get_program());

        params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0)}, kernel_selector::eltwise_mode::ASSIGN});
        params.inputs[0] = convert_data_tensor(impl_param.get_input_layout(), impl_param.input_offsets[0]);
        if (impl_param.get_program().get_node(primitive->id).is_dynamic()) {
            // WA to always match compiled dynamic kernel with dispatch data
            // W/O enforcing this option we may generate kernel for "broadcast" scneario due to umatched tensor dimensions
            // but in runtime dispatch data will be generated for non-broadcast case as shapes are actually same.
            params.broadcast = true;
        }
        return {params, optional_params};
    }
        void update_dispatch_data(const kernel_impl_params& impl_param) override {
            auto kernel_params = get_kernel_params(impl_param, true);
            (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
            update_kernels_list_to_skip();
    }
};

namespace detail {

attach_crop_impl::attach_crop_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32,
        data_types::i64
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<crop>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<crop>::create<crop_impl>,
                                     dyn_types,
                                     dyn_formats);

    implementation_map<crop>::add(impl_types::ocl, typed_primitive_impl_ocl<crop>::create<crop_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i64, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i64, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::f32, format::fyxb),
        std::make_tuple(data_types::f16, format::fyxb),
        std::make_tuple(data_types::i64, format::fyxb),
        std::make_tuple(data_types::i32, format::fyxb),
        std::make_tuple(data_types::i8, format::fyxb),
        std::make_tuple(data_types::u8, format::fyxb),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i64, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_zyx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i64, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::crop_impl)
