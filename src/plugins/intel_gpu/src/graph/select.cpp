// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "select_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include "ngraph/op/select.hpp"
#include "select_shape_inference.hpp"

namespace cldnn {
primitive_type_id select::type_id() {
    static primitive_type_base<select> instance;
    return &instance;
}

layout select_inst::calc_output_layout(select_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for select_node!");

    auto in_layout = impl_param.get_non_padded_input_layout(1);
#if 0
    auto in_layout = node.input(1).get_non_padded_output_layout();
    auto output_size = in_layout.get_tensor();

    if (impl_param.typed_desc<select>()->broadcast_type == "numpy") {
        auto input1_size = impl_param.get_input_layout(1).get_tensor();
        auto input2_size = impl_param.get_input_layout(2).get_tensor();
        output_size = tensor::max(input1_size, input2_size);
    }

    return layout(in_layout.data_type, in_layout.format, output_size);
#endif
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        node.get_dependency(0).get_output_layout().get_partial_shape(),
        node.get_dependency(1).get_output_layout().get_partial_shape(),
        node.get_dependency(2).get_output_layout().get_partial_shape()
    };

    ov::op::v1::Select op;
    ov::op::v1::shape_infer(&op, input_shapes, output_shapes);
    auto output_layout = layout{output_shapes[0], node.get_dependency(1).get_output_layout().data_type, node.get_dependency(1).get_output_layout().format};
    return output_layout;
}

std::string select_inst::to_string(select_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite select_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        select_info.add("input_" + std::to_string(i), node.input(i).id());
    }

    node_info->add("select info", select_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

select_inst::typed_primitive_inst(network& network, select_node const& node) : parent(network, node) {
    auto& deps = node.get_dependencies();

    CLDNN_ERROR_LESS_THAN(node.id(),
                                "Number of inputs",
                                deps.size(),
                                "Expected number of inputs",
                                3,
                                "");

    if (deps[1]->get_output_layout().get_tensor() != cldnn::tensor(1))
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Mask format",
                              deps[0]->get_output_layout().format,
                              "Positive input format",
                              deps[1]->get_output_layout().format,
                              "");

    if (deps[2]->get_output_layout().get_tensor() != cldnn::tensor(1))
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Mask format",
                              deps[0]->get_output_layout().format,
                              "Positive input format",
                              deps[2]->get_output_layout().format,
                              "");

    if (node.get_primitive()->broadcast_type == "none") {
        CLDNN_ERROR_LAYOUT_MISMATCH(node.id(),
                                "Positive input layout",
                                deps[1]->get_output_layout(),
                                "Negative input layout",
                                deps[2]->get_output_layout(),
                                "");

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Mask size",
                                deps[0]->get_output_layout().get_tensor(),
                                "Positive input format",
                                deps[1]->get_output_layout().get_tensor(),
                                "");
    } else if (node.get_primitive()->broadcast_type == "numpy") {
        if (deps[1]->get_output_layout().get_tensor() != cldnn::tensor(1) && deps[2]->get_output_layout().get_tensor() != cldnn::tensor(1))
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Positive input format",
                                  deps[1]->get_output_layout().format,
                                  "Negative input format",
                                  deps[2]->get_output_layout().format,
                                  "");

        CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                "Positive input data type",
                                deps[1]->get_output_layout().data_type,
                                "Negative input data type",
                                deps[2]->get_output_layout().data_type,
                                "");

        auto dep1_size = deps[1]->get_output_layout().get_tensor();
        auto dep2_size = deps[2]->get_output_layout().get_tensor();
        cldnn::tensor output_tensor = tensor::max(dep1_size, dep2_size);
        auto max_dim_count = output_tensor.raw.size();

        for (size_t i = 0; i < deps.size(); i++) {
            for (size_t d = 0; d < max_dim_count; d++) {
                auto current_dim = deps[i]->get_output_layout().get_tensor().raw[d];

                CLDNN_ERROR_BOOL(node.id(),
                                    "Sizes equal or broadcast is possible",
                                    !(current_dim == output_tensor.raw[d] || current_dim == 1),
                                    "Invalid input shapes");
            }
        }
    } else {
        CLDNN_ERROR_MESSAGE(node.id(), "Unsupported broadcast_type: " + node.get_primitive()->broadcast_type);
    }
}
}  // namespace cldnn
