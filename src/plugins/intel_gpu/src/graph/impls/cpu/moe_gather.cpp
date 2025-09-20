// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "moe_gather_inst.h"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace cpu {

struct moe_gather_impl : public typed_primitive_impl<moe_gather> {
    using parent = typed_primitive_impl<moe_gather>;
    using parent::parent;

    int num_active_experts = 0; 
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::moe_gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_gather_impl>(*this);
    }

    moe_gather_impl() : parent("moe_gather_cpu_impl") {}

    explicit moe_gather_impl(const moe_gather_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_gather>(), "[GPU] Incorrect program_node type");
        //const auto& node = arg.as<moe_gather>();
        //num_active_experts = node.get_primitive()->num_active_experts;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
//        ob << num_active_experts;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
//        ib >> num_active_experts;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_gather_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "moe_gather::execute_impl");
        auto& stream = instance.get_network().get_stream();

        if (instance.can_be_optimized()) {
            return stream.group_events(events);
        }

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();
        if (!pass_through_events) {
            stream.wait_for_events(events);
        }
        auto input_activations_mem_ptr = instance.dep_memory_ptr(0);
        auto gather_info_mem_ptr = instance.dep_memory_ptr(1);
        auto out_mem_ptr = instance.output_memory_ptr(0);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> input_data(input_activations_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::read> gather_info_data(gather_info_mem_ptr, stream);
        cldnn::mem_lock<ov::float16, mem_lock_type::read_write> output(out_mem_ptr, stream);

        auto params = instance.get_impl_params();
        const auto& desc = params->typed_desc<moe_gather>(); 
        auto num_total_experts = desc->num_total_experts;
        auto experts_data_offset_ptr = &gather_info_data[0];
        auto experts_data_num_ptr = experts_data_offset_ptr + num_total_experts;
        auto tokens_per_expert_ptr = experts_data_num_ptr + num_total_experts;

        auto hidden_size = instance.get_input_layout(0).get_shape()[1];

        size_t out_offset = 0;
        for (auto expert = 0; expert < num_total_experts; expert++) {
            auto expert_offset = experts_data_offset_ptr[expert];
            auto num_tokens_per_expert = experts_data_num_ptr[expert];
            if (expert_offset == -1) {
                continue;
            }
//            std::cout << "Read " << num_tokens_per_expert << " tokens from offset " << expert_offset << " for expert " << expert << std::endl;
            for (auto i = 0; i < num_tokens_per_expert; i++) {
                auto token = tokens_per_expert_ptr[expert_offset + i];
//                std::cout << "gathering token " << token << " for expert " << expert << std::endl;
                // copy input activation to output
                auto input_ptr = &input_data[token * hidden_size];
                for (size_t h = 0; h < hidden_size; h++) {
                    output[out_offset++] = input_ptr[h];
                }
            }
        }

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const moe_gather_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_gather_impl>();
    }
};


namespace detail {

attach_moe_gather_impl::attach_moe_gather_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<moe_gather>::add(impl_types::cpu, shape_types::static_shape, moe_gather_impl::create, types, formats);
    implementation_map<moe_gather>::add(impl_types::cpu, shape_types::dynamic_shape, moe_gather_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::moe_gather_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gather)
