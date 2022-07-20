// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include "ngraph/op/tile.hpp"
#include "tile_shape_inference.hpp"


namespace cldnn {
primitive_type_id tile::type_id() {
    static primitive_type_base<tile> instance;
    return &instance;
}

layout tile_inst::calc_output_layout(tile_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for tile_node!");
#if 1
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto input_format = input_layout.format;
    if (desc->output_shape_partial.get_shape().size() != 0) {
        return layout{input_layout.data_type, input_format, desc->output_shape_partial};
    } else {
        return layout{input_layout.data_type, input_format, desc->out_shape};
    }
#else
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        node.get_dependency(0).get_output_layout().size,
        node.get_dependency(1).get_output_layout().size,
    };

    ov::op::v0::Tile op;
    ov::op::v0::shape_infer(&op, input_shapes, output_shapes);
    auto output_layout = layout{node.get_dependency(0).get_output_layout().data_type, node.get_dependency(0).get_output_layout().format, output_shapes[0]};
    return output_layout;
#endif
}

std::string tile_inst::to_string(tile_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite tile_info;
    tile_info.add("input id", input.id());
    node_info->add("tile info", tile_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

tile_inst::typed_primitive_inst(network& network, tile_node const& node) : parent(network, node) {}

}  // namespace cldnn
