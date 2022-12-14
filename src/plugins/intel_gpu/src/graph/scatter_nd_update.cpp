// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scatter_nd_update)

layout scatter_nd_update_inst::calc_output_layout(scatter_nd_update_node const& node, kernel_impl_params const& impl_param) {

    auto desc = impl_param.typed_desc<scatter_nd_update>();

    auto input_layout = impl_param.get_input_layout();

    auto output_format = input_layout.format;
    auto output_shape = input_layout.get_partial_shape();
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    return layout{output_shape, output_type, output_format};

}

std::string scatter_nd_update_inst::to_string(scatter_nd_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_nd_update_info;
    scatter_nd_update_info.add("input id", input.id());

    node_info->add("scatter_nd_update info", scatter_nd_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_nd_update_inst::typed_primitive_inst(network& network, scatter_nd_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
