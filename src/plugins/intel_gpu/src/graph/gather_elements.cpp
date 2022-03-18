// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id gather_elements::type_id() {
    static primitive_type_base<gather_elements> instance;
    return &instance;
}

layout gather_elements_inst::calc_output_layout(gather_elements_node const& node) {
    auto op = node.get_primitive();

    auto data_sizes = node.input(0).get_output_layout().get_ordered_dims();
    auto indices_sizes = node.input(1).get_output_layout().get_ordered_dims();
    auto output_sizes = indices_sizes;
    auto output_format = cldnn::format::any;
    if (output_sizes.size() <= 4) {
        output_format = cldnn::format::bfyx;
    } else if (output_sizes.size() == 5) {
        output_format = cldnn::format::bfzyx;
    } else {
        output_format = cldnn::format::bfwzyx;
    }

    auto output_sizes_tensor = tensor(tensor(output_sizes).sizes(output_format));
    auto padding = op->output_padding;

    auto output_data_type = node.has_fused_primitives()
        ?node.get_fused_output_layout().data_type
        :node.input(0).get_output_layout().data_type;
    return layout(output_data_type, output_format, output_sizes_tensor, padding);
}

std::string gather_elements_inst::to_string(gather_elements_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_elements_info;
    gather_elements_info.add("input id", input.id());
    gather_elements_info.add("input shape", node.input(0).get_output_layout().size.to_string());
    gather_elements_info.add("indices shape", node.input(1).get_output_layout().size.to_string());
    gather_elements_info.add("target axis", desc->axis);
    gather_elements_info.add("output shape", calc_output_layout(node).size.to_string());

    node_info->add("gather_elements info", gather_elements_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_elements_inst::typed_primitive_inst(network& network, gather_elements_node const& node) : parent(network, node) {}

}  // namespace cldnn
