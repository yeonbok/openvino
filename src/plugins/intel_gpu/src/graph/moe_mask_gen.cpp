// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_mask_gen_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_mask_gen)

layout moe_mask_gen_inst::calc_output_layout(moe_mask_gen_node const& node, kernel_impl_params const& impl_param) {
    OPENVINO_THROW("moe_mask_gen has multiple outputs so only supports allow_new_shape_infer = true.");
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[1];
}

template<typename ShapeType>
std::vector<layout> moe_mask_gen_inst::calc_output_layouts(moe_mask_gen_node const& /*node*/, const kernel_impl_params& impl_param) {
    // TODO
    std::vector<layout> output_layouts;
    const auto& num_total_experts = impl_param.typed_desc<moe_mask_gen>()->num_total_experts;
    const auto& num_experts_per_token = impl_param.typed_desc<moe_mask_gen>()->num_experts_per_token;
    auto num_actual_used_experts_shape = ov::Shape{static_cast<size_t>(1)};
    // out0: num_actual_expert
    output_layouts.emplace_back(num_actual_used_experts_shape, data_types::i32, format::bfyx);
    if (impl_param.get_input_layout(0).is_dynamic()) {
        // out1: tokens_per_expert
        auto tokens_per_expert_shape = ov::PartialShape::dynamic();
        output_layouts.emplace_back(tokens_per_expert_shape, data_types::i32, format::bfyx);
    } else {
        const auto num_tokens = impl_param.get_input_layout(0).get_shape()[0];
        // out1: tokens_per_expert
        auto tokens_per_expert_shape = ov::Shape{num_tokens * num_experts_per_token};
        output_layouts.emplace_back(tokens_per_expert_shape, data_types::i32, format::bfyx);
    }
    // out2: experts_info_start_idx
    auto experts_info_start_idx_shape = ov::Shape{static_cast<size_t>(num_total_experts)};
     output_layouts.emplace_back(experts_info_start_idx_shape, data_types::i32, format::bfyx);
    // out3: experts_id
    auto experts_ids = ov::Shape{static_cast<size_t>(num_total_experts)};
    output_layouts.emplace_back(experts_ids, data_types::i32, format::bfyx);
    // out4: tokens_lens_per_expert
    auto tokens_lens_per_expert = ov::Shape{static_cast<size_t>(num_total_experts)};
    output_layouts.emplace_back(tokens_lens_per_expert, data_types::i32, format::bfyx);
    return output_layouts;
}

template std::vector<layout> moe_mask_gen_inst::calc_output_layouts<ov::PartialShape>(moe_mask_gen_node const& node, const kernel_impl_params& impl_param);

std::string moe_mask_gen_inst::to_string(moe_mask_gen_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    std::stringstream primitive_description;

    json_composite moe_mask_gen_info;
    if (desc->output_data_types[0].has_value())
        moe_mask_gen_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_mask_gen_inst::typed_primitive_inst(network& network, moe_mask_gen_node const& node) : parent(network, node) { }
}  // namespace cldnn
