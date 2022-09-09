// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "input_layout_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace {
bool has_optimized_users(input_layout_node const& node) {
    for (auto& user : node.get_users()) {
        if (user->can_be_optimized()) {
            return true;
        }
    }

    return false;
}
}  // namespace

namespace cldnn {
primitive_type_id input_layout::type_id() {
    static primitive_type_base<input_layout> instance;
    return &instance;
}

input_layout_node::typed_program_node(const std::shared_ptr<input_layout> dprim, program& prog)
    : parent(dprim, prog) {
    can_share_buffer(false);
}

input_layout_inst::typed_primitive_inst(network& network, input_layout_node const& node)
    : parent(network, node, !node.is_dynamic() && (!network.is_internal() || has_optimized_users(node))) {
    _has_valid_input = false;  // by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

void input_layout_inst::set_data(memory::ptr mem) {
    auto ol = node.get_output_layout();

    check_memory_to_set(*mem, ol);

    if (mem->is_allocated_by(get_network().get_engine())) {
        _output = mem;
    } else {
        mem_lock<char, mem_lock_type::read> src(mem, get_network().get_stream());
        mem_lock<char, mem_lock_type::write> dst(_output, get_network().get_stream());
        std::copy(src.begin(), src.end(), dst.begin());
    }

    _has_valid_input = true;
    _output_changed = true;
    _shape_changed = mem->get_layout() != ol;

    if (_shape_changed) {
        auto il_prim = std::const_pointer_cast<input_layout>(node.get_primitive());
        il_prim->change_layout(mem->get_layout());
        _impl_params->output_layout = mem->get_layout();
    }
}

void input_layout_inst::update_shape() {
    OPENVINO_ASSERT(_output != nullptr, "[GPU] input memory is not set");
    auto mem_layout = _output->get_layout();
    if (_impl_params->output_layout != mem_layout) {
        set_shape_change();
    }
    _impl_params->output_layout = mem_layout;
}

std::string input_layout_inst::to_string(input_layout_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
