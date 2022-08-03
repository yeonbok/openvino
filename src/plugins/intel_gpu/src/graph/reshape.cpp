// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "intel_gpu/primitives/reshape.hpp"
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
    auto prim = node.get_primitive();
    auto input_layout = node.input().get_non_padded_output_layout();

    if (input_layout.is_static()) {
        auto sizes = prim->output_shape;
        #if 0
        if (sizes.size() < input_layout.format.dimension()) {
            sizes.insert(sizes.end(), input_layout.format.dimension() - sizes.size(), 1);
        }
        #endif
        auto input_sizes = input_layout.get_dims();
        int64_t need_recalc = -1;
        ov::Dimension::value_type shape_count = 1;

        for (size_t i = 0; i < sizes.size(); i++) {
            if (sizes[i].is_dynamic()) {
                if (need_recalc >= 0) {
                    CLDNN_ERROR_MESSAGE(node.id(), "Only one dimension of the new shape can be -1");
                }
                need_recalc = i;
                continue;
            }
            // when output pattern is 0, then we need to copy corresponding dimension from input
            if (sizes[i] == 0) {
                sizes[i] = input_sizes[i];
            }
            shape_count *= sizes[i].get_length();
        }
        if (need_recalc >= 0)
            sizes[need_recalc] = static_cast<int>(input_layout.count()) / shape_count;

        node.reset_shape_ready();

        return layout{sizes, input_layout.data_type, input_layout.format};
    } else {
        return layout{prim->output_shape, input_layout.data_type, input_layout.format};
    }
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

void reshape_inst::update_shape() {
    if (!_network.shape_changed())
        return;

    auto& node = const_cast<reshape_node&>(dynamic_cast<const reshape_node&>(_node));

    if (_node.get_dependencies().size() == 2) {
        auto in_node = _node.get_dependency(1).id();
        auto shape_mem = _network.get_output_memory(in_node);
        // TODO: usm_device is copied to host on lock(), but we need to ensure that this is better, then
        // keeping such constants on host (i.e. modifying transfer_memory_to_device)
        // if (shape_mem->get_allocation_type() == allocation_type::usm_device) {
        //     IE_THROW() << " lockable memory is required to update shape for reshape prim\n";
        // }
        auto reshape_prim = std::static_pointer_cast<reshape>(std::const_pointer_cast<primitive>(_node.get_primitive()));
        if (_network.has_event(in_node)) {
            const auto& ev = _network.get_primitive_event(in_node);
            _network.get_stream().wait_for_events({ev});
        }
        reshape_prim->output_shape = ov::PartialShape(read_vector<size_t>(shape_mem, _network.get_stream()));
        node.set_shape_ready();
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    // TODO: modify kernel_impl_param with dyn_layout
    auto new_layout = _node.type()->calc_output_layout(_node, *_node.get_kernel_impl_params());
    auto out_layout = _node.is_valid_output_layout() ? _node.get_output_layout() : layout(data_types::f32, format::any, tensor{});
    auto out_layout_str = _node.is_valid_output_layout() ? out_layout.to_string() : "invalid";
    GPU_DEBUG_IF(debug_config->verbose >= 4) {
        GPU_DEBUG_COUT << id() << " update shape: was: " << out_layout_str << " now: " << new_layout.to_string() << std::endl;
    }
    if (!_node.is_valid_output_layout() || _node.get_output_layout() != new_layout)
        set_shape_change();
    // TODO: Get rid of this const_cast
    node.set_output_layout(new_layout);
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
