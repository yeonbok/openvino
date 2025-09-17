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
    // TODO
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> moe_mask_gen_inst::calc_output_layouts(moe_mask_gen_node const& /*node*/, const kernel_impl_params& impl_param) {
    // TODO
    const auto& num_total_experts = impl_param.typed_desc<moe_mask_gen>()->num_total_experts;
//    const auto& num_active_experts = impl_param.typed_desc<moe_mask_gen>()->num_active_experts;
    std::vector<layout> output_layouts;
    auto gather_info_shape = ov::Shape{static_cast<size_t>(num_total_experts * 2)};
    auto gemm_info_shape = ov::Shape{static_cast<size_t>(num_total_experts * 6)};
    output_layouts.emplace_back(gather_info_shape, data_types::i32, format::bfyx);
    output_layouts.emplace_back(gemm_info_shape, data_types::i32, format::bfyx);
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

moe_mask_gen_inst::typed_primitive_inst(network& network, moe_mask_gen_node const& node) : parent(network, node, true) { }
}  // namespace cldnn
