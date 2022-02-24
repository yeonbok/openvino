// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "gather_shape_inference.hpp"
#include "openvino/op/gather.hpp"

namespace cldnn {
primitive_type_id gather::type_id() {
    static primitive_type_base<gather> instance;
    return &instance;
}

layout gather_inst::calc_output_layout(gather_node const& node) {
    auto desc = node.get_primitive();
    auto input_layout = node.input(0).get_output_layout();
    auto output_type = input_layout.data_type;

    {
        auto output_format = desc->output_format;

        ov::op::v8::Gather op;
        std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
        std::vector<ov::PartialShape> input_shapes = {
            node.get_dependency(0).get_output_layout().size,
            node.get_dependency(1).get_output_layout().size,
            ov::PartialShape{1} // axis input is removed on gather primitive creation, so we can't use get_dependency(2)
        };

        int64_t axis = desc->axis;

        auto axis_tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {{2, axis_tensor}};
        ov::op::util::shape_infer(&op, input_shapes, output_shapes, const_data);
        return layout{output_type, output_format, output_shapes[0]};
    }

    auto shape = desc->output_shape.to_shape();
    std::vector<tensor::value_type> dims_converted(shape.begin(), shape.end());
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++)
        dims_converted.push_back(1);

    format output_format = input_layout.format;
    if (dims_converted.size() == 5) {
        switch (input_layout.format) {
        case format::bfyx:
            output_format = format::get_default_format(dims_converted.size());
            break;
        case format::b_fs_yx_fsv16:
            output_format = format::b_fs_zyx_fsv16;
            break;
        case format::b_fs_yx_fsv32:
            output_format = format::b_fs_zyx_fsv32;
            break;
        case format::bs_fs_yx_bsv16_fsv16:
            output_format = format::bs_fs_zyx_bsv16_fsv16;
            break;
        default:
            break;
        }
    } else if (dims_converted.size() == 6) {
        switch (input_layout.format) {
        case format::bfyx:
        case format::bfzyx:
            output_format = format::get_default_format(dims_converted.size());
            break;
        default:
            break;
        }
    }

    return layout{output_type,
                  output_format,
                  tensor(format::get_default_format(dims_converted.size()), dims_converted)};
}

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", cldnn::to_string(desc->output_shape));

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_inst::typed_primitive_inst(network& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
