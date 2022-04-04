// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "convolution/convolution_kernel_selector.h"
#include "convolution/convolution_params.h"
#include <algorithm>

namespace cldnn {
namespace ocl {

struct deformable_conv_impl : typed_primitive_impl_ocl<deformable_conv> {
    using parent = typed_primitive_impl_ocl<deformable_conv>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_conv_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<deformable_conv>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        args.bias = instance.bias_term() ? instance.bias_memory(split) : nullptr;
        return args;
    }

    int32_t get_split() const override { return _outer.get_split(); }

    uint32_t get_groups() const override { return _outer.get_groups(); }

public:
    static primitive_impl* create(const deformable_conv_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& weights_layout = arg.weights(0).get_output_layout();
        const auto& weights_size = weights_layout.size;

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;

        const auto depthwise_separable_opt = arg.get_depthwise_sep_opt();
        const auto actual_split = depthwise_separable_opt ? (decltype(split))1 : split;

        std::vector<layout> input_layouts;
        for (auto i : arg.get_dependencies()) {
            input_layouts.push_back(i->get_output_layout());
        }

        const auto& bias_layout = arg.bias_term() ?  arg.bias().get_output_layout() : layout(data_types::f32, format::any, tensor());
        kernel_impl_params param_info = kernel_impl_params(arg.get_program().get_id(), arg.get_unique_id(), arg.id(),
                                                           arg.get_primitive()->type_string(), input_layouts, arg.get_output_layout(),
                                                           arg.get_program(), arg.get_fused_primitives(),
                                                           arg.get_fused_activations_funcs(), arg.get_fused_activations_params(),
                                                           weights_layout, arg.bias_term(), bias_layout);

        auto conv_params = get_weights_bias_default_params<kernel_selector::convolution_params>(
            param_info,
            (groups > 1 && !depthwise_separable_opt) ? groups : actual_split,
            groups);
        auto conv_optional_params =
            get_default_weights_bias_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        conv_params.depthwise_separable_opt = depthwise_separable_opt;
        conv_params.split = split;
        conv_params.groups = groups;
        conv_params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        auto& kernel_selector = kernel_selector::deformable_conv_kernel_selector::Instance();
        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto conv = new deformable_conv_impl(arg, best_kernels[0]);

        return conv;
    }
};

struct deformable_interp_impl : typed_primitive_impl_ocl<deformable_interp> {
    using parent = typed_primitive_impl_ocl<deformable_interp>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<deformable_interp_impl>(*this);
    }

protected:
    int32_t get_split() const override { return 1; }

    uint32_t get_groups() const override { return 1; }

public:
    static primitive_impl* create(const deformable_interp_node& arg) {
        const auto& primitive = arg.get_primitive();
        const auto& input_layout = arg.input().get_output_layout();
        const auto& kernel_size = primitive->kernel_size;

        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& pad = primitive->pad;
        const auto& groups = primitive->groups;
        const auto& deformable_groups = primitive->deformable_groups;

        std::vector<layout> input_layouts;
        for (auto i : arg.get_dependencies()) {
            input_layouts.push_back(i->get_output_layout());
        }

        kernel_impl_params param_info = kernel_impl_params(arg.get_program().get_id(), arg.get_unique_id(), arg.id(),
                                                           arg.get_primitive()->type_string(), input_layouts, arg.get_output_layout(),
                                                           arg.get_program(), arg.get_fused_primitives(),
                                                           arg.get_fused_activations_funcs(), arg.get_fused_activations_params());

        auto conv_params = get_default_params<kernel_selector::convolution_params>(param_info, groups);
        auto conv_optional_params =
            get_default_optional_params<kernel_selector::convolution_optional_params>(arg.get_program());

        // It's not really needed, just initialize fields of params
        auto weights_layout = layout(input_layout.data_type, input_layout.format, kernel_size);
        conv_params.weights = convert_weights_tensor(weights_layout);

        conv_params.inputs.push_back(convert_data_tensor(arg.trans().get_output_layout()));
        if (primitive->input.size() == 3) {
            conv_params.inputs.push_back(convert_data_tensor(arg.mask().get_output_layout()));
            conv_params.deformable_mask_enabled = true;
        }
        conv_params.bilinear_interpolation_pad = primitive->bilinear_interpolation_pad;
        conv_params.deformable_groups = deformable_groups;

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);

        conv_params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        conv_params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;

        conv_params.dilation = {dilation_x, dilation_y, dilation_z};

        conv_params.kernelSize = { (uint32_t)kernel_size.spatial[0],
                                   (uint32_t)kernel_size.spatial[1],
                                   (uint32_t)kernel_size.spatial[2] };

        auto& kernel_selector = kernel_selector::deformable_interp_kernel_selector::Instance();
        kernel_selector::KernelsData best_kernels = kernel_selector.GetBestKernels(conv_params, conv_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with these arguments");
        auto conv = new deformable_interp_impl(arg, best_kernels[0]);

        return conv;
    }
};

namespace detail {

attach_deformable_conv_impl::attach_deformable_conv_impl() {
    implementation_map<deformable_conv>::add(impl_types::ocl, deformable_conv_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

attach_deformable_interp_impl::attach_deformable_interp_impl() {
    implementation_map<deformable_interp>::add(impl_types::ocl, deformable_interp_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
