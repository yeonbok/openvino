// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {

primitive_type_id reshape::type_id() {
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto input_layout = node.input().get_non_padded_output_layout();
    auto sizes = node.get_primitive()->output_shape.sizes();
    auto input_sizes = input_layout.size.sizes();
    size_t need_recalc = 0;
    uint32_t shape_count = 1;

    for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i] == -1) {
            if (need_recalc) {
                CLDNN_ERROR_MESSAGE(node.id(), "Only one dimension of the new shape can be -1");
            }
            need_recalc = i;
            continue;
        }
        if (sizes[i] == 0) {
            sizes[i] = input_sizes[i];
        }
        shape_count *= sizes[i];
    }
    if (need_recalc)
        sizes[need_recalc] = static_cast<int>(input_layout.size.count()) / shape_count;

    input_layout.size = tensor(sizes);
    return input_layout;
}

std::string reshape_inst::to_string(reshape_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite reshape_info;
    reshape_info.add("input id", input.id());
    reshape_info.add("output shape", desc->output_shape);

    node_info->add("reshape info", reshape_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reshape_inst::typed_primitive_inst(network& network, reshape_node const& node) : parent(network, node, false) {
    std::cout << "reshape inst created!!!!!!!!!!!!!!" << std::endl;
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Input layout data typr",
                                    input_layout.data_type,
                                    "output layout data type",
                                    output_layout.data_type,
                                    "");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Output layout count",
                          output_layout.count(),
                          "input layout count",
                          input_layout.count(),
                          "Output layout of reshape primitive changes size of input buffer");
}

void reshape_inst::allocate_memories() {
    allocate_output();
}

memory::ptr reshape_inst::allocate_output() {
    // if reshape operated in-place, postpone creation of the output until network run,
    // then create new memory object as the reinterpreted output of the previous primitive
    if (!node.can_be_optimized()) {
        std::cout << "allocate reshape output!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        _output = primitive_inst::allocate_output();
        return _output;
    } else {
        std::cout << " reusing reshape memory!" << std::endl;
        reuse_input();
        return _output;
    }
}

void reshape_inst::on_execute() {
    std::cout << "reshape => on_execute()" << std::endl;
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void reshape_inst::reuse_input() {
    build_deps();  // reshape need deps
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}

}  // namespace cldnn
