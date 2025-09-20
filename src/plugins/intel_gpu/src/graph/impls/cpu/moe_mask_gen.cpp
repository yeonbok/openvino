// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "moe_mask_gen_inst.h"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace cpu {

struct moe_mask_gen_impl : public typed_primitive_impl<moe_mask_gen> {
    using parent = typed_primitive_impl<moe_mask_gen>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::moe_mask_gen_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_mask_gen_impl>(*this);
    }

    moe_mask_gen_impl() : parent("moe_mask_gen_cpu_impl") {}

    explicit moe_mask_gen_impl(const moe_mask_gen_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_mask_gen>(), "[GPU] Incorrect program_node type");
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_mask_gen_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "moe_mask_gen::execute_impl");
        auto& stream = instance.get_network().get_stream();

        if (instance.can_be_optimized()) {
            return stream.group_events(events);
        }

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

//        auto params = instance.get_impl_params();

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        // [seqlen:30, active_expert:2]
//        std::cout << "topk layout " << instance.get_input_layout(0).to_short_string() << std::endl;
        auto num_tokens = instance.get_input_layout(0).get_shape()[0];
        auto num_active_experts = instance.get_node().as<moe_mask_gen>().get_primitive()->num_active_experts;
        auto num_total_experts = instance.get_node().as<moe_mask_gen>().get_primitive()->num_total_experts;

        auto topk_idx_mem_ptr = instance.dep_memory_ptr(0);
//        auto topk_weight_mem_ptr = instance.dep_memory_ptr(1);
        auto gather_info_mem_ptr = instance.output_memory_ptr(0);
//        auto gemm_info_mem_ptr = instance.output_memory_ptr(1);

        cldnn::mem_lock<int32_t, mem_lock_type::read> topk_idx_lock(topk_idx_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::read_write> gather_info_lock(gather_info_mem_ptr, stream);
//        cldnn::mem_lock<uint8_t, mem_lock_type::read_write> gemm_info_lock(gemm_info_mem_ptr, stream);

        auto topk_idx_ptr = topk_idx_lock.data();
        auto gather_info_ptr = gather_info_lock.data();

        auto expert_data_offset_ptr = &gather_info_ptr[0];
        auto expert_data_num_ptr = expert_data_offset_ptr + num_total_experts;
        auto tokens_per_expert_ptr = expert_data_num_ptr + num_total_experts;
//        auto gemm_info_ptr = reinterpret_cast<int32_t*>(gemm_info_lock.data());
        // make mask for gather
        std::vector<std::vector<int32_t>> tokens_per_expert(num_total_experts, std::vector<int32_t>());
        for (size_t token = 0; token < num_tokens; ++token) {
            for (int j = 0; j < num_active_experts; ++j) {
                auto expert_id = topk_idx_ptr[token * num_active_experts + j];
                tokens_per_expert[expert_id].push_back(token);
            }
        }

        // gather_info_ptr : pack two information 
        // 1st half : offset of each expert experts_offset{num_total_experts}
        // 2nd half : tokens per each expert
        // e.g., 4 total experts / 2 active experts tokens_per_experts{num_total_tokens * num_active_experts}
        // where
        //    - token 0  : uses exp0, exp1
        //    - token 1  : uses exp1, exp2 
        // first half : experts_offset : [0, 1, 3, -1]
        // second half : tokens_per_expert : [[0], [0, 1], [1], []]
        int expert_offset = 0;
        for (int expert = 0; expert < num_total_experts ; expert++) {
            // store offset
            if (tokens_per_expert[expert].empty()) {
                expert_data_offset_ptr[expert] = -1;
            } else {
                expert_data_offset_ptr[expert] = expert_offset;
            }
            // store num of tokens for each expert
            expert_data_num_ptr[expert] = static_cast<int32_t>(tokens_per_expert[expert].size());
            // recollect tokens for each expert
            for (int token : tokens_per_expert[expert]) {
                tokens_per_expert_ptr[expert_offset] = token;
//                std::cout << "wrote token " << token << " for expert " << expert << " at offset " << expert_offset << std::endl;
                expert_offset++;
            }
        }
        // make mask for gemm
        // TODO

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_mem_ptrs[i]->unlock(stream);

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const moe_mask_gen_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_mask_gen_impl>();
    }
};


namespace detail {

attach_moe_mask_gen_impl::attach_moe_mask_gen_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::i32,
        data_types::i64,
        data_types::f32,
    };

    implementation_map<moe_mask_gen>::add(impl_types::cpu, shape_types::static_shape, moe_mask_gen_impl::create, types, formats);
    implementation_map<moe_mask_gen>::add(impl_types::cpu, shape_types::dynamic_shape, moe_mask_gen_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::moe_mask_gen_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_mask_gen)
