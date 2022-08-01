// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "reshape_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

#include "shape_nodes.hpp"

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
    auto input_sizes = input_layout.get_tensor().sizes();
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
        sizes[need_recalc] = static_cast<int>(input_layout.count()) / shape_count;

    return layout{input_layout.data_type, input_layout.format, tensor(sizes)};
}

std::vector<layout> reshape_inst::calc_output_layouts(reshape_node const& node, const kernel_impl_params& impl_param) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for reshape_node!");
    auto prim = node.get_primitive();
    auto input_layout = node.input().get_non_padded_output_layout();

    ov::op::v1::Reshape op;
    op.set_special_zero(prim->special_zero);

    ov::PartialShape pattern_shape = node.get_dependencies().size() == 2 ? node.get_dependency(1).get_output_layout().get_partial_shape()
                                                                         : ov::Shape{ prim->output_pattern.size() };
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        node.get_dependency(0).get_output_layout().get_partial_shape(),
        pattern_shape,
    };

    auto& constant_mem = impl_param.memory_deps;

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

static std::vector<int64_t> read_vector(cldnn::memory::ptr mem, cldnn::stream& stream) {
    std::vector<int64_t> out_vecs;
    if (mem->get_allocation_type() == allocation_type::usm_host || mem->get_allocation_type() == allocation_type::usm_shared) {
        switch (mem->get_layout().data_type) {
            case data_types::i32: {
                int32_t* p_mem = reinterpret_cast<int32_t*>(mem->buffer_ptr());
                for (int i=0; i < mem->count(); i++) {
                    out_vecs.push_back(p_mem[i]);
                }
                break;
            }
            case data_types::i64: {
                int64_t* p_mem = reinterpret_cast<int64_t*>(mem->buffer_ptr());
                for (int i=0; i < mem->count(); i++) {
                    out_vecs.push_back(p_mem[i]);
                }
                break;
            }
            default: IE_THROW() << "read_vector: unsupported data type";
        }
    } else {
        switch (mem->get_layout().data_type) {
            case data_types::i32: {
                mem_lock<int32_t, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<int64_t>(lock.begin(), lock.end()));
            }
            case data_types::i64: {
                mem_lock<int64_t, mem_lock_type::read> lock{mem, stream};
                out_vecs = std::move(std::vector<int64_t>(lock.begin(), lock.end()));
            }
            default: IE_THROW() << "read_vector: unsupported data type";
        }
    }
    return out_vecs;
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
        reshape_prim->output_shape = ov::PartialShape(read_vector(shape_mem, _network.get_stream()));
        node.set_shape_ready();
    }

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto new_layout = _node.type()->calc_output_layout(_node);
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
