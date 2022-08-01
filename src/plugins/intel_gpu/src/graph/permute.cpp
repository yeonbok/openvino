// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "permute_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"

#include <algorithm>
#include <string>
#include <vector>

namespace cldnn {

primitive_type_id permute::type_id() {
    static primitive_type_base<permute> instance;
    return &instance;
}

layout permute_inst::calc_output_layout(permute_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for permute_node!");
    auto input_layout = node.input().get_output_layout();
    auto permute_order = node.get_primitive()->permute_order;
    ov::PartialShape output_shape;

    auto input_shape = input_layout.size;

    for (size_t x = 0; x < permute_order.size(); x++) {
        output_shape.push_back(input_shape[permute_order[x]]);
    }

    auto op = node.get_primitive()->output_padding;

    if (node.has_fused_primitives()) {
        input_layout.data_type = node.get_fused_output_layout().data_type;
    }

    return layout(input_layout.data_type, input_layout.format, output_shape, op);
}

std::string permute_inst::to_string(permute_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto permute_order = desc->permute_order;
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_permute_order;

    for (size_t i = 0; i < permute_order.size(); ++i) {
        ss_permute_order << permute_order.at(i);
        i != (permute_order.size() - 1) ? ss_permute_order << ", " : ss_permute_order << "";
    }

    json_composite permute_info;
    permute_info.add("input id", input.id());
    permute_info.add("permute order", ss_permute_order.str());

    node_info->add("permute info", permute_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

permute_inst::typed_primitive_inst(network& network, permute_node const& node) : parent(network, node) {
    auto permute_order = argument.permute_order;

    auto required_order_values_size = static_cast<uint32_t>(permute_order.size());

    for (decltype(required_order_values_size) i = 0; i < required_order_values_size; i++) {
        if (!(std::find(permute_order.begin(), permute_order.end(), i) != permute_order.end()))
            CLDNN_ERROR_MESSAGE(node.id(), "Permute order does not contain all of required values.");
    }
}
}  // namespace cldnn
