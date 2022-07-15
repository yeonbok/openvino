// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>

#include "shape_nodes.hpp"

namespace cldnn {

primitive_type_id reshape::type_id() {
    static primitive_type_base<reshape> instance;
    return &instance;
}

layout reshape_inst::calc_output_layout(reshape_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto input_layout = impl_param.get_non_padded_input_layout();
    auto desc = impl_param.typed_desc<reshape>();
    auto sizes = desc->output_shape.sizes();
    auto input_sizes = input_layout.get_tensor().sizes();
    size_t need_recalc = 0;
    uint32_t shape_count = 1;

    for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i] == -1) {
            if (need_recalc) {
                CLDNN_ERROR_MESSAGE(desc->id, "Only one dimension of the new shape can be -1");
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
        sizes[need_recalc] = static_cast<int>(input_layout.count()) / shape_count;

    return layout{input_layout.data_type, input_layout.format, tensor(sizes)};
}

std::vector<layout> reshape_inst::calc_output_layouts(reshape_node const& node, kernel_impl_params const& impl_param,
                                                      const std::map<int, memory::ptr> constant_mem) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto prim = node.get_primitive();
    auto input_layout = node.input().get_non_padded_output_layout();

    ov::op::v1::Reshape op;
    op.set_special_zero(prim->special_zero);

    ov::PartialShape pattern_shape = node.get_dependencies().size() == 2 ? node.get_dependency(1).get_output_layout().get_partial_shape()
                                                                         : ov::Shape{ prim->output_pattern.size() };

    auto input_shape = node.get_dependency(0).get_output_layout().get_partial_shape();
    if (input_shape.is_dynamic()) {
        auto rank = pattern_shape.rank().get_length();
        return { layout{ov::PartialShape::dynamic(rank),
                        input_layout.data_type,
                        input_layout.format.adjust_to_rank(rank)} };
    }


    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        input_shape,
        pattern_shape,
    };

    if (!constant_mem.empty()) {
        auto pattern_mem = constant_mem.at(1);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> pattern_lock(pattern_mem, node.get_program().get_stream());

        auto pattern_ptr = pattern_lock.data();
        auto pattern_tensor = make_host_tensor(pattern_mem->get_layout(), pattern_ptr);

        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, pattern_tensor},
        };

        shape_infer(&op, input_shapes, output_shapes, const_data);
    } else {
        auto pattern_data = prim->output_pattern;
        auto pattern_tensor = make_host_tensor({pattern_shape, data_types::i64, format::bfyx}, static_cast<void*>(pattern_data.data()));
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, pattern_tensor},
        };

        shape_infer(&op, input_shapes, output_shapes, const_data);
    }

    return { layout{output_shapes[0], input_layout.data_type, input_layout.format.adjust_to_rank(output_shapes[0].size())} };
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
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(),
                                    "Input layout data typr",
                                    input_layout.data_type,
                                    "output layout data type",
                                    output_layout.data_type,
                                    "");
    if (output_layout.is_static())
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output layout count",
                              output_layout.count(),
                              "input layout count",
                              input_layout.count(),
                              "Output layout of reshape primitive changes size of input buffer");

    // if reshape operated in-place, postpone creation of the output until network run,
    // then create new memory object as the reinterpreted output of the previous primitive
    if (_node.get_output_layout().is_static()) {
        if (!node.can_be_optimized())
            _output = allocate_output();
        else
            reuse_input();
    } else {
        if (_exec_deps.size() > 0 && input_memory_ptr())
            reuse_input();
    }
}

void reshape_inst::on_execute() {
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
