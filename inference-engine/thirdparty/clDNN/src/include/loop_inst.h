/*
// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include "api/loop.hpp"
#include "api/mutable_data.hpp"
#include "api/input_layout.hpp"
#include "api/memory.hpp"

#include "network_impl.h"
#include "primitive_inst.h"
#include "error_handler.h"
#include "engine_impl.h"
#include "../src/gpu/ocl_queue_wrapper.h"
#include "../src/gpu/ocl_base_event.h"
#include <string>
#include <memory>
#include <vector>
#include "error_handler.h"

namespace cldnn {
template<>
struct typed_program_node<loop> : public typed_program_node_base<loop> {
private:
    using parent = typed_program_node_base<loop>;
    topology body_topology;
    topology_impl& body;

    std::vector<loop::io_primitive_map> input_primitive_maps;
    std::vector<loop::io_primitive_map> output_primitive_maps;
    std::vector<cldnn::loop::backedge_mapping> back_edges;
    bool use_current_iteration;
    bool use_execution_condition;
    mutable program_impl::ptr body_program;
    mutable std::map<primitive_id, memory_impl::ptr> backedge_mem_impls;
    mutable std::map<primitive_id, std::shared_ptr<mutable_data>> backedge_layers;
    mutable std::map<primitive_id, std::shared_ptr<memory>> backedge_mem;

    mutable bool output_is_backedge;

    void setup_internal_mutabledata_node(primitive_id md_id, layout md_layout, std::vector<primitive_id> md_inputs_id = {}, uint32_t net_id = 0) const {
        if (body.get_primitives().count(md_id) == 0) {
            backedge_mem_impls[md_id] = get_program().get_engine().allocate_memory(md_layout, net_id, false);
            backedge_mem[md_id] = std::make_shared<memory>(backedge_mem_impls[md_id].get());
            backedge_layers[md_id] = std::make_shared<mutable_data>(md_id, md_inputs_id, *backedge_mem[md_id]);
            body.add(backedge_layers[md_id]);
        }
    }

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) :
        parent(prim, prog),
        body_topology(this->get_primitive()->body),
        body(*body_topology.get()),
        input_primitive_maps(this->get_primitive()->input_primitive_maps),
        output_primitive_maps(this->get_primitive()->output_primitive_maps),
        back_edges(this->get_primitive()->back_edges),
        use_current_iteration(!this->get_primitive()->current_iteration_id.empty()),
        use_execution_condition(!this->get_primitive()->condition_id.empty()),
        max_iteration(this->get_primitive()->max_iteration < 0 ? DEFAULT_MAX_NUM_ITERATION : this->get_primitive()->max_iteration) {}

    mutable size_t iteration_axis;
    int32_t max_iteration;

    int32_t get_max_iteration() const { return max_iteration; }
    program_impl::ptr get_body_program() const { return body_program; }
    bool is_output_working_as_backedge() const { return output_is_backedge; }
    bool is_current_iteration_used() const { return use_current_iteration; }
    bool is_execution_condition_used() const { return use_execution_condition; }

    std::vector<const loop::io_primitive_map*> find_io_primitive_maps(
                                                            const primitive_id& prim_id,
                                                            bool is_external) const {
        std::vector<const loop::io_primitive_map*> ret;
        if (is_external) {
            for (const auto& it : input_primitive_maps) {
                if (it.external_id == prim_id) {
                    ret.push_back(&it);
                }
            }
            for (const auto& it : output_primitive_maps) {
                if (it.external_id == prim_id) {
                    ret.push_back(&it);
                }
            }
        } else {
            for (const auto& it : input_primitive_maps) {
                if (it.internal_id == prim_id) {
                    ret.push_back(&it);
                }
            }
            for (const auto& it : output_primitive_maps) {
                if (it.internal_id == prim_id) {
                    ret.push_back(&it);
                }
            }
        }
        return ret;
    }

    static size_t convert_to_raw_axis(const int axis, const int ndim) {
        // convert between bfyx, bfzyx, bfzyxw and tensor.size.raw
        assert(axis < ndim);
        if (axis < 2) {
            return axis;
        }
        return (ndim - 1) - (axis - 2);
    }

    layout calc_body_input_layout(const loop::io_primitive_map& inputDesc) const {
        const auto& dependency_list = this->get_dependencies();
        auto input = std::find_if(dependency_list.begin(), dependency_list.end(), [&inputDesc](const program_node* p){
            return p->id() == inputDesc.external_id;
        });
        assert(input != dependency_list.end());
        layout calculated_layout = (*input)->get_output_layout();
        auto shape = calculated_layout.size.sizes(calculated_layout.format);

        if (inputDesc.axis >= 0) {
            iteration_axis = convert_to_raw_axis(inputDesc.axis, static_cast<int>(shape.size()));
            calculated_layout.size.raw[iteration_axis] = 1; // cropped inputs shape
        }

        return calculated_layout;
    }

    const std::vector<loop::io_primitive_map>& get_input_primitive_maps() const { return input_primitive_maps; }
    const std::vector<loop::io_primitive_map>& get_output_primitive_maps() const { return output_primitive_maps; }

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID, bool external_id = true) {
        if (external_id) {
            for (auto& pm : input_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                }
            }
        } else {
            for (auto& pm : input_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                }
            }
            for (auto& pm : output_primitive_maps) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                }
            }
            for (auto& back_edge : back_edges) {
                if (back_edge.from == prevID) {
                    back_edge.from = newID;
                }
                if (back_edge.to == prevID) {
                    back_edge.to = newID;
                }
            }
        }
    }

    const std::vector<cldnn::loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    static bool is_integer(const data_types& data_type) {
        switch (data_type) {
            case data_types::i8:
            case data_types::i32:
            case data_types::i64:
                return true;
            default:
                return false;
        }
    }

    void process_single_int_input(const primitive_id& id) const {
        const topology_map& body_topology_map = body.get_primitives();
        if (!id.empty()) {
            // add input_layout if not exist
            if (body_topology_map.count(id)) {
                layout body_input_layout(data_types::i32, format::bfyx, {1, 1, 1, 1});
                body.add(std::make_shared<input_layout>(id, body_input_layout));
            } else {
                const auto& body_input_prim = body.at(id);
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    body_input_prim->type != input_layout::type_id(),
                    id + " is not cldnn::input_layout");
                const auto input_layout_prim = static_cast<const input_layout*>(body_input_prim.get());
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    !static_cast<bool>(input_layout_prim->output_data_type),
                    "data_type of " + id + " is not specified");
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    !is_integer(*input_layout_prim->output_data_type),
                    id + " is not integer type");
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    input_layout_prim->layout.count() != 1,
                    id + " should have 1 element");
            }
        }
    }

    void build_body_program() const {
        const std::vector<cldnn::program_node *>& deps = get_dependencies();
        // setup internal inputs
        const primitive_id& trip_count_id = get_trip_count_id();
        const primitive_id& initial_execution = get_initial_execution_id();
        const primitive_id& num_iteration = get_num_iteration_id();
        for (const cldnn::program_node * dep : deps) {
            const primitive_id& id = dep->id();
            if (id == trip_count_id || id == initial_execution || id == num_iteration) {
                continue;
            }

            for (const auto& pm : input_primitive_maps) {
                layout calculated_layout = calc_body_input_layout(pm);
                const primitive_id& internal_input_id = pm.internal_id;

                // add inputs for body network if not exist
                if (body.get_primitives().count(internal_input_id) == 0) {
                    body.add(std::make_shared<input_layout>(internal_input_id, calculated_layout));
                } else {
                    body.change_input_layout(internal_input_id, calculated_layout);
                }
            }
        }

        // setup internal output
        // TODO: handle multiple output_primitive_map
        std::set<primitive_id> output_names;
        output_names.insert(output_primitive_maps.begin()->internal_id);
        const auto& back_edges_list = this->get_primitive()->back_edges;

        // add current_iteration_id in body network, condition_id if exist
        process_single_int_input(get_current_iteration_id());
        process_single_int_input(get_condition_id());

        // setup outputs for backedges
        for (auto& back_edge : back_edges_list) {
            // check whether the back_edge.to has its corresponding io_primitive_map
            const auto& input_map = std::find_if(input_primitive_maps.begin(), input_primitive_maps.end(),
                [&](const loop::io_primitive_map& pm) {
                    return pm.internal_id == back_edge.to;
                });
            if (input_map == input_primitive_maps.end()) {
                std::string msg = "No primitive mapping for backedge (internal_id: " + back_edge.to + ')';
                CLDNN_ERROR_MESSAGE(this->id(), msg.c_str());
            }

            for (const auto& prim : body.get_primitives()) {
                if (prim.first != back_edge.from) {
                    continue;
                }
                const auto dependencies_ref = prim.second->dependencies();
                std::vector<primitive_id> dep_pids(dependencies_ref.size());
                for (const auto& dep : dependencies_ref) {
                    dep_pids.emplace_back(dep.get());
                }
                setup_internal_mutabledata_node(back_edge.from, calc_body_input_layout(*input_map), dep_pids);
            }

            output_names.insert(back_edge.from);
        }

        auto opts = get_program().get_options();
        std::vector<primitive_id> output_names_vec(output_names.begin(), output_names.end());
        opts.set_option(build_option::outputs(output_names_vec));
        body_program = get_program().get_engine().build_program(body, opts, false);
    }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->initial_execution_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->current_iteration_id; }
    const primitive_id& get_condition_id() const { return get_primitive()->condition_id; }
    const primitive_id& get_num_iteration_id() const { return get_primitive()->num_iteration_id; }
    const topology& get_body_topology() const { return get_primitive()->body; }
};

using loop_node = typed_program_node<loop>;

template <>
class typed_primitive_inst<loop> : public typed_primitive_inst_base<loop> {
    using parent = typed_primitive_inst_base<loop>;

public:
    struct backedge_memory_mapping {
        enum backedge_type {
            // output memory(from_primitive) of body network needs to be concatenated
            CONCAT_OUTPUT,
            // output memory(from_primitive) of body network does not need to be concateneated
            SINGLE,
        };
        std::shared_ptr<primitive_inst> from_primitive;
        std::shared_ptr<primitive_inst> to_primitive;
        std::vector<memory_impl::ptr> from_mems;
        memory_impl::ptr initial_mem;
        backedge_type type;
        size_t total_bytes;

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> from_primitive, std::shared_ptr<primitive_inst> to_primitive,
            std::vector<memory_impl::ptr> from_mems, memory_impl::ptr initial_mem, backedge_type type = CONCAT_OUTPUT):
            from_primitive(from_primitive),
            to_primitive(to_primitive),
            from_mems(from_mems),
            type(type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        backedge_memory_mapping(
            std::shared_ptr<primitive_inst> from_primitive, std::shared_ptr<primitive_inst> to_primitive,
            memory_impl::ptr from_mem, memory_impl::ptr initial_mem, backedge_type type = SINGLE):
            from_primitive(from_primitive),
            to_primitive(to_primitive),
            from_mems{from_mem},
            initial_mem(initial_mem),
            type(type),
            total_bytes(initial_mem->get_layout().bytes_count()) {
                validate_backedge_memory();
            }

        void setup_iteration(int64_t iter) const {
            if (type == CONCAT_OUTPUT) {
                if (iter == 0) {
                    to_primitive->set_output_memory(*initial_mem);
                } else if (iter > 0) {
                    to_primitive->set_output_memory(*from_mems.at(iter - 1));
                } else {
                    throw std::runtime_error("Invalid iteraton count" + std::to_string(iter));
                }
            } else {
                if (iter == 0) {
                    copy_initial_data();
                }
            }
        }

private:
        void validate_backedge_memory() {
            for (const auto& from_mem : from_mems) {
                const size_t from_mem_bytes = from_mem->get_layout().bytes_count();
                if (from_mem_bytes != total_bytes) {
                    throw std::runtime_error("Invalid backedge memory layout: "
                        "size not matched with that of initial_mem");
                }
            }
        }

        void copy_initial_data() const {
            mem_lock<uint8_t> from_lock {initial_mem};
            mem_lock<uint8_t> to_lock {from_mems.front()};
            const auto src = from_lock.begin();
            const auto dst = to_lock.begin();
            std::copy(src, src + total_bytes, dst);
        }
    };

    struct concatenated_memory_mapping {
        concatenated_memory_mapping(
                             network_impl::ptr body_network,
                             const primitive_id& concat_data_id,
                             const primitive_id& sliced_data_id,
                             memory_impl::ptr concatenated_mem,
                             std::vector<memory_impl::ptr> sliced_mems,
                             int iteration_elements = 0,
                             int stride = 0,
                             int initial_offset = 0) :
            concat_data_id(concat_data_id),
            sliced_data_id(sliced_data_id),
            concatenated_mem(concatenated_mem),
            sliced_mems(sliced_mems),
            bytes_per_element(static_cast<int>(data_type_traits::size_of(concatenated_mem->get_layout().data_type))),
            bytes_per_iteration(iteration_elements * bytes_per_element),
            bytes_stride(stride * bytes_per_element),
            bytes_initial_offset(initial_offset * bytes_per_element),
            body_network(body_network) {
//               assert(concatenated_mem->get_allocation_type() == allocation_type::usm_shared);
               concatenated_mem_ptr = concatenated_mem->lock();
               concatenated_mem->unlock();

               sliced_mem_ptrs.reserve(sliced_mems.size());
               for (const auto& sliced_mem : sliced_mems) {
//                   assert(sliced_mem->get_allocation_type() == allocation_type::usm_shared);
                   sliced_mem_ptrs.push_back(sliced_mem->lock());
                   sliced_mem->unlock();
               }
            }

        void restore_concatenated_mem() const {
            mem_lock<uint8_t> concat_mem_lock{ concatenated_mem };
            int output_offset = bytes_initial_offset;
            for (const auto& sliced_mem : sliced_mems) {
                mem_lock<uint8_t> sliced_mem_lock{ sliced_mem };
                uint8_t* src = sliced_mem_lock.data();
                uint8_t* dst = concat_mem_lock.data() + output_offset;
                std::copy(src, src + bytes_per_iteration, dst);
                output_offset += bytes_stride;
            }
        }

        void setup_concatenated_output_memory(uint64_t iteration) const {
            const auto& sliced_output_mem = sliced_mems.at(iteration);
            concat_data_prim->set_output_memory(*sliced_output_mem);
        }

        void copy_sliced_input_mem(int iteration) const {
            const int offset = bytes_initial_offset + bytes_stride * iteration;
            mem_lock<uint8_t> from_lock{ concatenated_mem };
            mem_lock<uint8_t> to_lock{ sliced_mems.at(iteration) };
            const auto src = from_lock.begin() + offset;
            const auto dst = to_lock.begin();
            std::copy(src, src + bytes_per_iteration, dst);
        }

        memory_impl::ptr get_sliced_mem(int iteration) const {
            return sliced_mems.at(iteration);
        }

        std::vector<cl::Event> get_cl_event(const std::vector<event_impl::ptr>& deps) const {
//            assert(!body_network->get_engine().get_context()->get_configuration().host_out_of_order);
            std::vector<cl::Event> dep_events;
            for (auto& dep : deps) {
                if (auto ocl_base_ev = dynamic_cast<gpu::ocl_base_event*>(dep.get())) {
                    dep_events.push_back(ocl_base_ev->get());
                }
            }
            // if (!context()->get_configuration().host_out_of_order) {
            //     for (auto& dep : deps) {
            //         if (auto ocl_base_ev = dynamic_cast<ocl_base_event*>(dep.get())) {
            //             dep_events.push_back(ocl_base_ev->get());
            //         }
            //     }
            // } else {
            //     dep_events_ptr = nullptr;

            //     sync_events(deps);
            // }
            return dep_events;
        }

        event_impl::ptr enqueue_sliced_input_memcpy(int iteration, const std::vector<event_impl::ptr>& events) const {
            // direction = 0: concatenated memory to sliced memory
            return enqueue_usm_memcpy(iteration, events, 0);
        }

        event_impl::ptr enqueue_concatenated_output_memcpy(int iteration, const std::vector<event_impl::ptr>& events) const {
            // direction = 1: sliced memory to concatenated memory
            return enqueue_usm_memcpy(iteration, events, 1);
        }

        event_impl::ptr enqueue_usm_memcpy(int iteration, const std::vector<event_impl::ptr>& events, int direction) const {
            assert(direction == 0 || direction == 1);

            std::shared_ptr<cldnn::gpu_toolkit> ctx = body_network->get_engine().get_context();
            const uint32_t queue_id = body_network->get_id();
            const cl::CommandQueue& cmd_queue = ctx->queue(queue_id);

            const int offset = bytes_initial_offset + bytes_stride * iteration;
            // concatenated memory to sliced memory (direction = 0)
            void* src = reinterpret_cast<uint8_t*>(concatenated_mem_ptr) + offset;
            void* dst = sliced_mem_ptrs.at(iteration);
            if (direction == 1) {
                // sliced memory to concatenated memory
                std::swap(src, dst);
            }
            // TODO(eunsoo): blocking should be true?
            std::vector<cl::Event> dep_events = get_cl_event(events);
            cl::Event ret_event;
            cl::usm::enqueue_memcpy(cmd_queue, dst, src, bytes_per_iteration, false, &dep_events, &ret_event);
            return ctx->ocl_event(queue_id, ret_event);
        }

        primitive_id concat_data_id;
        primitive_id sliced_data_id;
        std::shared_ptr<primitive_inst> concat_data_prim;
        std::shared_ptr<primitive_inst> sliced_data_prim;
        memory_impl::ptr concatenated_mem;
        std::vector<memory_impl::ptr> sliced_mems;
        const int bytes_per_element;
        const int bytes_per_iteration;
        const int bytes_stride;
        const int bytes_initial_offset;
        network_impl::ptr body_network;
        void* concatenated_mem_ptr;
        std::vector<void*> sliced_mem_ptrs;
    };

    static layout calc_output_layout(const loop_node& node);
    bool preproc_memories_done;
    std::vector<backedge_memory_mapping> backedge_memory_mappings;
    std::vector<concatenated_memory_mapping> concatenated_input_mem_mappings;
    std::vector<concatenated_memory_mapping> concatenated_output_mem_mappings;

    static std::string to_string(const loop_node& node);

public:
    typed_primitive_inst(network_impl& network, const loop_node& node);
    network_impl::ptr get_body_network() const { return body_network; }

private:
    network_impl::ptr body_network;
};

using loop_inst = typed_primitive_inst<loop>;
}  // namespace cldnn
