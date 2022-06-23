// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <limits>

namespace cldnn {
primitive_type_id arg_max_min::type_id() {
    static primitive_type_base<arg_max_min> instance;
    return &instance;
}

layout arg_max_min_inst::calc_output_layout(arg_max_min_node const& node, int32_t idx) {
    // TODO(taylor) : temporarily set same as first output type. TBD for different second ouutput dtype
    auto desc = node.get_primitive();
    auto input_layout = node.get_dependency(0).first->get_output_layout();
    bool values_first = desc->values_first;
    data_types output_data_type;
    data_types output_idx_type;
    output_data_type = desc->output_data_types.size() > 0 ? *desc->output_data_types[0] : input_layout.data_type;
    output_idx_type = desc->output_data_types.size() > 1 ? *desc->output_data_types[1] : *(desc->output_data_types[0]);
    auto size_check = [&](size_t tensor_size) {
        if (desc->input.size() == 1 && values_first)
            return;
        size_t max_size;
        // lowest integer not representable in floating point type = 2^(mantissa_bits + 1) + 1
        // https://stackoverflow.com/questions/3793838/which-is-the-first-integer-that-an-ieee-754-float-is-incapable-of-representing-e
        if (output_idx_type == data_types::f32) {
            max_size = (1 << std::numeric_limits<float>::digits);
        } else if (output_idx_type == data_types::f16) {
            // mantissa_bits for fp16 = 10
            max_size = (1 << 11);
        } else if (output_idx_type == data_types::u8) {
            max_size = std::numeric_limits<uint8_t>::max();
        } else if (output_idx_type == data_types::i32) {
            max_size = std::numeric_limits<int32_t>::max();
        } else {
            max_size = std::numeric_limits<size_t>::max();
        }

        if (tensor_size > max_size) {
            CLDNN_ERROR_GREATER_THAN(node.id(),
                                     "Reduced tensor size",
                                     tensor_size,
                                     "Maximum output data type value",
                                     max_size,
                                     "Current output data type is unable to hold maximum index of a tensor.");
        }
    };
    auto format = input_layout.format;
    if (desc->with_axis) {
        switch (desc->axis) {
            case arg_max_min::x:
                size_check(input_layout.size.spatial[0]);
                if (format == cldnn::format::bfzyx)
                    return layout{output_data_type,
                                  format::bfzyx,
                                  tensor{input_layout.size.batch[0],
                                         input_layout.size.feature[0],
                                         (int32_t)desc->top_k,
                                         input_layout.size.spatial[1],
                                         input_layout.size.spatial[2]}};
                else
                    return layout{output_data_type,
                                  format,
                                  tensor{input_layout.size.batch[0],
                                         input_layout.size.feature[0],
                                         (int32_t)desc->top_k,
                                         input_layout.size.spatial[1]}};
            case arg_max_min::y:
                size_check(input_layout.size.spatial[1]);
                if (format == cldnn::format::bfzyx)
                    return layout{output_data_type,
                                  format::bfzyx,
                                  tensor{input_layout.size.batch[0],
                                         input_layout.size.feature[0],
                                         input_layout.size.spatial[0],
                                         (int32_t)desc->top_k,
                                         input_layout.size.spatial[2]}};
                else
                    return layout{output_data_type,
                                  format,
                                  tensor{input_layout.size.batch[0],
                                         input_layout.size.feature[0],
                                         input_layout.size.spatial[0],
                                         (int32_t)desc->top_k}};
            case arg_max_min::feature:
                size_check(input_layout.size.feature[0]);
                if (format == cldnn::format::bfzyx)
                    return layout{output_data_type,
                                  format::bfzyx,
                                  tensor{input_layout.size.batch[0],
                                         (int32_t)desc->top_k,
                                         input_layout.size.spatial[0],
                                         input_layout.size.spatial[1],
                                         input_layout.size.spatial[2]}};
                else
                    return layout{output_data_type,
                                  format,
                                  tensor{input_layout.size.batch[0],
                                         (int32_t)desc->top_k,
                                         input_layout.size.spatial[0],
                                         input_layout.size.spatial[1]}};
            case arg_max_min::batch:
                size_check(input_layout.size.batch[0]);
                if (format == cldnn::format::bfzyx)
                    return layout{output_data_type,
                                  format::bfzyx,
                                  tensor{(int32_t)desc->top_k,
                                         input_layout.size.feature[0],
                                         input_layout.size.spatial[0],
                                         input_layout.size.spatial[1],
                                         input_layout.size.spatial[2]}};
                else
                    return layout{output_data_type,
                                  format,
                                  tensor{(int32_t)desc->top_k,
                                         input_layout.size.feature[0],
                                         input_layout.size.spatial[0],
                                         input_layout.size.spatial[1]}};
            case arg_max_min::z:
                size_check(input_layout.size.spatial[2]);
                return layout{output_data_type,
                              format::bfzyx,
                              tensor{input_layout.size.batch[0],
                                     input_layout.size.feature[0],
                                     input_layout.size.spatial[0],
                                     input_layout.size.spatial[1],
                                     (int32_t)desc->top_k}};
            default:
                break;
        }
    }
    size_check(input_layout.size.feature[0] * input_layout.size.spatial[0] * input_layout.size.spatial[1]);
    return layout{output_data_type,
                  input_layout.format,
                  tensor{input_layout.size.batch[0], 1, (int32_t)desc->top_k, 1}};
}
#if 0 // TODO(taylor)
std::string arg_max_min_inst::to_string(arg_max_min_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto axis = desc->with_axis ? "true" : "false";
    auto out_type = desc->output_type ? "max" : "min";

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("top_k", desc->top_k);
    conv_info.add("with axis", axis);
    if (desc->with_axis)
        conv_info.add("axis", desc->axis);
    conv_info.add("output type", out_type);
    node_info->add("arg_max_min info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
#endif
arg_max_min_inst::typed_primitive_inst(network& network, arg_max_min_node const& node) : parent(network, node) {}
}  // namespace cldnn
