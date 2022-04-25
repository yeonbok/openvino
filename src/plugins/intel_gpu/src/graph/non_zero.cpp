// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "non_zero_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {

// -----------------------------------------------
// count_nonzero
// -----------------------------------------------
primitive_type_id count_nonzero::type_id() {
    static primitive_type_base<count_nonzero> instance;
    return &instance;
}

layout count_nonzero_inst::calc_output_layout(count_nonzero_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for count_nonzero_node!");
    return layout{cldnn::data_types::i32, cldnn::format::bfyx, tensor{1,1,1,4}};
}

std::string count_nonzero_inst::to_string(count_nonzero_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite count_nonzero_info;
    count_nonzero_info.add("input id", input.id());
    count_nonzero_info.add("output shape", tensor{1,1,1,4});

    node_info->add("count_nonzero info", count_nonzero_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

count_nonzero_inst::typed_primitive_inst(network& network, count_nonzero_node const& node) : parent(network, node) {}

static std::vector<int64_t> read_vector(cldnn::memory::ptr mem, cldnn::stream& stream) {
    switch (mem->get_layout().data_type) {
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        default: IE_THROW() << "read_vector: unsupported data type";
    }
}

// -----------------------------------------------
// gather_nonzero
// -----------------------------------------------
primitive_type_id gather_nonzero::type_id() {
    static primitive_type_base<gather_nonzero> instance;
    return &instance;
}

layout gather_nonzero_inst::calc_output_layout(gather_nonzero_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for gather_nonzero_node!");
    auto prim = node.get_primitive();
    return layout{cldnn::data_types::i32, cldnn::format::bfyx, prim->output_shape};
}

std::string gather_nonzero_inst::to_string(gather_nonzero_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_nonzero_info;
    gather_nonzero_info.add("input id", input.id());
    gather_nonzero_info.add("output shape", desc->output_shape);

    node_info->add("gather_nonzero info", gather_nonzero_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_nonzero_inst::typed_primitive_inst(network& network, gather_nonzero_node const& node) : parent(network, node, false) {}

void gather_nonzero_inst::update_shape() {
    auto& node = const_cast<gather_nonzero_node&>(dynamic_cast<const gather_nonzero_node&>(_node));
    if (_node.get_dependencies().size() == 2) {
        auto shape_mem = _network.get_output_memory(_node.get_dependency(1).id());
        auto gather_nonzero_prim = std::static_pointer_cast<gather_nonzero>(std::const_pointer_cast<primitive>(_node.get_primitive()));
        gather_nonzero_prim->output_shape = ov::PartialShape(read_vector(shape_mem, _network.get_stream()));
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
    
    node.set_output_layout(new_layout);
}

}  // namespace cldnn
