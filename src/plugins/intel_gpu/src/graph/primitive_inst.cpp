// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "generic_layer_inst.h"
#include "input_layout_inst.h"
#include "arg_max_min_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>
#include <stack>
#include <vector>
#include <memory>
#include <algorithm>


#if 0
#define PRINT_TIME(func) \
{ \
 auto start = std::chrono::high_resolution_clock::now(); \
 func; \
 auto duration = std::chrono::high_resolution_clock::now() - start; \
 std::cerr << id() << " " <<  #func <<  " " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() << "us\n"; \
}
#else
#define PRINT_TIME(func) func;
#endif

namespace {

bool is_optimized_output_user(const program_node* user) {
    if (user->can_be_optimized()) {
        if (user->is_output())
            return true;

        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_optimized_output_user(u)) {
                return true;
            }
        }
        return false;
    }
    return false;
}

bool is_output_buffer(const program_node& node) {
    if (node.is_output())
        return true;

    // Try to recursively find any optimized out user which is also network output
    for (const auto& user : node.get_users()) {
        if (is_optimized_output_user(user)) {
            return true;
        }
    }

    return false;
}

}  // namespace

namespace cldnn {

bool is_user_cpu(const program_node* user) {
    if (user->can_be_optimized()) {
        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_user_cpu(u)) {
                return true;
            }
        }
        return false;
    }
    if (auto impl = user->get_selected_impl())
        return impl->is_cpu();
    return false;
}

bool is_any_user_cpu(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (is_user_cpu(user))
            return true;
    }
    return false;
}

uint32_t primitive_inst::get_network_id() const { return _network.get_id(); }

void primitive_inst::update_shape() {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    if (!_network.shape_changed())
        return;

    std::map<size_t, memory::ptr> memory_deps;
    for (auto& i : _node.get_shape_infer_dependencies()) {
        auto& dep = _node.get_dependency(i);
        if (dep.is_type<data>()) {
            memory_deps.insert({i, dep.as<data>().get_attached_memory_ptr()});
            continue;
        }
        auto dep_id = dep.id();
        if (_network.has_event(dep.id())) {
            const auto& ev = _network.get_primitive_event(dep_id);
            _network.get_stream().wait_for_events({ev});
            GPU_DEBUG_IF(debug_config->verbose >= 4) {
                GPU_DEBUG_COUT << id() << " wait for " << i << " dependency\n";
            }
        }
        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
    }

    layout new_layout = layout(data_types::f32, format::bfyx, tensor());
    auto params = _node.get_kernel_impl_params();
    params->memory_deps = memory_deps;
    auto out_layouts = _node.type()->calc_output_layouts(_node, *params);
    if (out_layouts.empty())
        new_layout = _node.type()->calc_output_layout(_node);
    else
        new_layout = out_layouts[0];

    auto out_layout = _node.is_valid_output_layout() ? _node.get_output_layout() : layout(data_types::f32, format::any, tensor{});
    auto out_layout_str = _node.is_valid_output_layout() ? out_layout.to_string() : "invalid";
    GPU_DEBUG_IF(debug_config->verbose >= 4) {
        GPU_DEBUG_COUT << id() << " update shape: was: " << out_layout_str << " now: " << new_layout.to_string() << std::endl;
    }

    if (out_layout != new_layout)
        set_shape_change();

    // TODO: Get rid of this const_cast
    const_cast<program_node&>(_node).set_output_layout(new_layout);
}

void primitive_inst::realloc_if_needed() {
    GPU_DEBUG_GET_INSTANCE(debug_config);

    if (!_output
        || ((_output->get_layout().count() < _node.get_output_layout().count())
        && (max_output_layout_size < _node.get_output_layout().count()))) {
        GPU_DEBUG_IF(debug_config->verbose >= 4) {
            GPU_DEBUG_COUT << "realloc memory for node: " << id() << std::endl;
        }
        _output = allocate_output();
    } else {
        _output = _network.get_engine().reinterpret_buffer(*_output, _node.get_output_layout());
    }
    max_output_layout_size = std::max(_output->get_layout().count(), max_output_layout_size);
}

void primitive_inst::update_impl() {
    auto pid = id();
    if (!_node.is_type<data>() && !(_node.is_type<mutable_data>() && _node.get_dependencies().empty())) {
        auto get_layout_key = [&]()->std::string {
            std::string layout_key_str = "";
            if (_node.is_valid_output_layout()) {
                layout_key_str = id() + "_" + std::to_string(_node.get_unique_id());
                layout_key_str += "_" + _node.get_output_layout().to_string();

                for (auto in : _node.get_dependencies()) {
                    if (!in->is_constant()) {
                        layout_key_str += "_" + in->get_output_layout().to_string();
                    }
                }
            }
            return layout_key_str;
        };

        auto layout_key = get_layout_key();
        if (layout_key != "") {
            auto cache = _network.get_program()->get_primitive_impl_cache();
            bool is_hit = false;
            std::shared_ptr<cldnn::primitive_impl> origin_impl;
            PRINT_TIME(std::tie(origin_impl, is_hit) = cache->get(layout_key, [&]() {
                cldnn::LRUCache<std::string, std::shared_ptr<cldnn::primitive_impl>>::CacheEntry new_entry;
                auto new_impl = std::move(_node.type()->choose_impl(_node));
                PRINT_TIME(_network.get_program()->compile());
                PRINT_TIME(new_impl->init_kernels());
                new_entry.size = 1;
                new_entry.data = std::move(new_impl);
                return new_entry;
            }));

            _impl = std::move(origin_impl->clone());
            if (is_hit) {
                PRINT_TIME(_impl->init_kernels());
            }
        } else {
            PRINT_TIME(_impl = std::move(_node.type()->choose_impl(_node)));
            PRINT_TIME(_network.get_program()->compile());
            PRINT_TIME(_impl->init_kernels());
        }

        reset_shape_change();
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 4) {
            GPU_DEBUG_COUT << "Update impl for node " << id() << ": " << (_impl != nullptr ? _impl->get_kernel_name() : "Nullptr") << std::endl;
        }
    }
}

void primitive_inst::check_memory_to_set(const memory& mem, const layout& layout) const {
    CLDNN_ERROR_LAYOUT_MISMATCH("network layout",
        "set memory layout",
        mem.get_layout(),
        "expected layout",
        layout,
        "");

    // check shared image/buffer compatibility, if applicable
    auto params = mem.get_internal_params();
    if (params.mem_type != shared_mem_type::shared_mem_empty) {
        if (!mem.is_allocated_by(get_network().get_engine())) {
            CLDNN_ERROR_MESSAGE(_node.id(), "Memory object is not suitable");
        }

        switch (params.mem_type) {
        case shared_mem_type::shared_mem_vasurface:
        case shared_mem_type::shared_mem_image:
            if (!layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output image instead of a buffer");
            break;
        case shared_mem_type::shared_mem_buffer:
        case shared_mem_type::shared_mem_dxbuffer:
            if (layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output buffer instead of an image");
            break;
        case shared_mem_type::shared_mem_usm:
            break;
        default:
            CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output memory of unknown/invalid type");
            break;
        }
    }
}

void primitive_inst::set_output_memory(memory::ptr mem_new, bool check) {
    auto& eng = _network.get_engine();
    // skip all the buzz if no action actually required
    if (_output && eng.is_the_same_buffer(*mem_new, *_output)) {
        return;
    }

    auto ol = _node.get_output_layout();

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (_node.is_constant()) {
        mem_new->copy_from(_network.get_stream(), *_output);
    }

    _output = mem_new;
}

event::ptr primitive_inst::execute(const std::vector<event::ptr>& events) {
    const auto primitive_id = id();
    CLDNN_ERROR_BOOL(primitive_id,
                     "Invalid/unset input",
                     !_has_valid_input,
                     "Cannot execute primitive " + primitive_id + " with invalid/unset input");

    static std::mutex m;
    {
        // Lock for program nodes
        // To be removed once concurrency issue for program node is resolved
        std::lock_guard<std::mutex> lock(m);
        PRINT_TIME(update_shape());
        if (shape_changed()) {
            PRINT_TIME(update_impl());
            PRINT_TIME(update_weights());
            PRINT_TIME(realloc_if_needed());
        }
    }

    if (_node.get_output_layout().is_dynamic())
        throw ov::Exception("Runtime output layout for node " + id() + " is dynamic!");

    on_execute();

    if (!_impl)
        throw ov::Exception("Impl for " + id() + " is nullptr");

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 1) {
        GPU_DEBUG_COUT << "Execute " << id() << ", memory type: "
                       << output_memory().get_allocation_type() << std::endl;
    }

    // If a node has mutable input or it's an output, then the input/output buffers might be changed
    // So we need to set arguments on each execution.
    // if (has_mutable_input() || is_output() || is_dynamic()) {
        set_arguments();
    // }

    if (_exec_deps.empty())
        return _impl->execute(events, *this);

    std::vector<event::ptr> dependencies;
    auto queue_type = get_network().get_stream().get_queue_type();
    if (queue_type == queue_types::out_of_order) {
        dependencies.reserve(_exec_deps.size());
        for (auto& input : _exec_deps) {
            auto id = input->id();
            try {
                // if the requested event does not exists it means that it has not been executed, so the processing_order is
                // wrong or synchronization failed.
                auto ev = get_network().get_primitive_event(id);
                dependencies.emplace_back(ev);
            } catch (const std::out_of_range& oor) {
                std::string temp = std::string("internal CLDNN error: execution order corrupted.") + std::string("\n") +
                                std::string(oor.what() + std::string("\n"));
                CLDNN_ERROR_MESSAGE(id, temp);
            }
        }
    }
    return _impl->execute(dependencies, *this);
}

void primitive_inst::set_arguments() {
    const auto primitive_id = id();
    CLDNN_ERROR_BOOL(primitive_id,
                     "Invalid/unset input",
                     !_has_valid_input,
                     "Cannot set arguments for primitive " + primitive_id + " with invalid/unset input");

    _impl->set_arguments(*this);
}

void primitive_inst::build_deps() {
    if (_deps.empty() && !_node.get_dependencies().empty()) {
        _deps = _network.get_primitives(_node.get_dependencies());
        _exec_deps = build_exec_deps(_deps);
    }
}

primitive_inst::primitive_inst(network& network, program_node const& node, bool allocate_memory)
    : _network(network), _node(node), _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr),
      _output(), _output_changed(false), _mem_allocated(allocate_memory) {
    if (allocate_memory) {
        // In case when output is mutable_data primitive, and other users dependencies are only used for
        // suychronization, The output memory of such primitive will be fused with mutable_data
        auto users = node.get_users();
        auto user_count = users.size();
        uint32_t mutable_data_count = 0;
        for (auto& user : users) {
            // Get mutable_data nodes count from nodes users
            if (user->is_type<mutable_data>()) {
                mutable_data_count++;
            }
        }

        // TODO: Remove WA for arg_max_min node.
        // For now it's required to handle the case when only second output of TopK primitive is used in plugin,
        // but kernels always write both outputs to the same memory object which leads to wrong result.
        if (user_count == 1 && mutable_data_count == 1 && !node.is_type<arg_max_min>()
                                                       && !node.is_type<experimental_detectron_roi_feature_extractor>()) {
            for (auto& user : node.get_users())
                if (user->is_type<mutable_data>())
                    _output = user->as<mutable_data>().get_attached_memory_ptr();
        } else {
            _output = allocate_output();
        }
    }
    if (_output)
        max_output_layout_size = _output->get_layout().count();
}

void primitive_inst::allocate_internal_buffers(void) {
    if (_impl == nullptr) return;
    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    if (ibuf_layouts.empty()) return;

    auto device_mem_acc = [&](size_t a, std::shared_ptr<primitive_inst> b) {
        if (!b->mem_allocated()) return a;
        if (b->output_memory().get_allocation_type() == allocation_type::usm_device ||
            b->output_memory().get_allocation_type() == allocation_type::cl_mem)
            return a + b->output_memory().size();
        else
            return a;
    };

    auto& engine = get_network().get_engine();
    bool input_device_mem = false;

    // NOTE: Currently the ocl driver aborts at runtime when there are layers using device memory close to max size within multiple streams.
    // Decided the limitation as 85 % empirically, but still it needs further investigation.
    const auto& inst_deps = _network.get_primitives(_node.get_dependencies());

    auto total_device_mem_size = std::accumulate(inst_deps.begin(), inst_deps.end(), 0, device_mem_acc);
    if (_output->get_allocation_type() ==  allocation_type::usm_device) {
        total_device_mem_size += _output->size();
    }

    int64_t available_device_mem_size = engine.get_device_info().max_global_mem_size - total_device_mem_size;
    // check if there is any device mem input
    if (engine.supports_allocation(allocation_type::usm_device)) {
        for (const auto& dep : inst_deps) {
            if (dep->output_memory().get_allocation_type() == allocation_type::usm_device) {
                input_device_mem = true;
                break;
            }
        }
    }

    for (auto layout : ibuf_layouts) {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": internal buf]" << std::endl;
        }
        if (input_device_mem && (available_device_mem_size - (int64_t)layout.bytes_count() >= 0))
            _intermediates_memory.push_back(engine.allocate_memory(layout, allocation_type::usm_device));
        else
            _intermediates_memory.push_back(engine.allocate_memory(layout, allocation_type::usm_host));
    }
}
memory::ptr primitive_inst::allocate_output(engine& _engine, memory_pool& pool, const program_node& _node, uint32_t net_id,
        bool is_internal) {
    auto get_memory_from_pool = [&](engine& _engine, const layout& layout, const primitive_id id, std::set<primitive_id> dependencies,
            allocation_type type, bool reusable) {
        if (_engine.configuration().use_memory_pool)
                return pool.get_memory(layout, id, net_id, dependencies, type, reusable);
        return pool.get_memory(layout, type);
    };

    auto layout = _node.get_output_layout();
    // TODO: Add a preprocessing step to do  alloc_type check before actual allocation
    const auto& node_deps = _node.get_dependencies();
    auto device_mem_acc = [&](size_t a, program_node* b) {
        return a + b->get_output_layout().bytes_count();
    };

    bool usm_device_allocatable = true;
    const auto& total_device_input_mem_size = std::accumulate(node_deps.begin(), node_deps.end(), (uint64_t)0, device_mem_acc);
    if (total_device_input_mem_size > _engine.get_device_info().max_global_mem_size)
        usm_device_allocatable = false;

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    bool is_cpu = _node.get_selected_impl() ? _node.get_selected_impl()->is_cpu() : false;
    auto use_lockable_memory = is_output_buffer(_node) || is_cpu || is_any_user_cpu(_node.get_users()) ||
                               !_engine.supports_allocation(allocation_type::usm_device);
    GPU_DEBUG_GET_INSTANCE(debug_config);
    const auto& lockable_mem_type = _engine.get_lockable_preffered_memory_allocation_type(layout.format.is_image_2d());
    const auto& alloc_type = use_lockable_memory ? lockable_mem_type
        : usm_device_allocatable ? allocation_type::usm_device : lockable_mem_type;

    if (is_internal && (_node.can_be_optimized() || _node.is_type<generic_layer>())) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                _node.get_memory_dependencies(),
                alloc_type,
                false);
    } else if (is_internal && _node.is_output() && _node.is_type<generic_layer>() &&
            _engine.supports_allocation(allocation_type::usm_device) && usm_device_allocatable) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return _engine.allocate_memory(layout, allocation_type::usm_device, false);
    } else if (is_internal && !_node.is_output() && _node.is_type<input_layout>()) {
        // Skip memory reset for input_layout primitives, since data will be copied from cldnn::data primitive
        // or just reuse primitive's memory
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": constant]" << std::endl;
        }
        return _engine.allocate_memory(layout, alloc_type, false);
    } else if (is_internal || (!_node.can_share_buffer()) || _node.can_be_optimized() || _node.is_output()) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return _engine.allocate_memory(layout, alloc_type);
    } else {
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                _node.get_memory_dependencies(),
                alloc_type,
                true);
    }
}
memory::ptr primitive_inst::allocate_output() {
    return allocate_output(get_network().get_engine(), _network.get_memory_pool(), _node, get_network_id(), _network.is_internal());
}

std::vector<std::shared_ptr<primitive_inst>> primitive_inst::build_exec_deps(
    std::vector<std::shared_ptr<primitive_inst>> const& deps) {
    std::vector<std::shared_ptr<primitive_inst>> exec_deps;
    exec_deps.reserve(deps.size());
    for (auto& dep : deps)
        if (dep->get_impl() != nullptr)
            exec_deps.push_back(dep);

    return exec_deps;
}

std::string primitive_inst::generic_to_string(program_node const& node, const char* type_name) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_inputs;

    for (size_t i = 0; i < node.get_dependencies().size(); ++i) {
        auto& in = node.get_dependency(i);
        ss_inputs << in.id();
        ss_inputs << ", count: " << in.get_output_layout().count();
        i != (node.get_dependencies().size() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    json_composite generic_info;
    generic_info.add("type_name", type_name);
    generic_info.add("deps count", node.get_dependencies().size());
    generic_info.add("deps", ss_inputs.str());

    node_info->add("generic info", generic_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
