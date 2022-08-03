// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

namespace cldnn {
primitive_type_id shape_of::type_id() {
    static primitive_type_base<shape_of> instance;
    return &instance;
}

layout shape_of_inst::calc_output_layout(shape_of_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<shape_of>();

    data_types dt = data_types::i32;

    if (prim->output_data_type)
        dt = *prim->output_data_type;

    if (impl_param.has_fused_primitives()) {
        dt = impl_param.get_fused_output_layout().data_type;
    }

    return layout{ov::PartialShape{static_cast<int64_t>(prim->output_rank)}, dt, format::bfyx};
}

std::string shape_of_inst::to_string(shape_of_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite shape_of_info;
    shape_of_info.add("out dt: ", dt_to_str(*desc->output_data_type));
    node_info->add("shape_of info", shape_of_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

shape_of_inst::typed_primitive_inst(network& network, shape_of_node const& node) : parent(network, node) { }
}  // namespace cldnn
