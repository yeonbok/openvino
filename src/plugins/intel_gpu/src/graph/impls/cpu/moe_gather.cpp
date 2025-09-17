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
        const auto& node = arg.as<moe_gather>();
        num_active_experts = node.get_primitive()->num_active_experts;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << num_active_experts;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> num_active_experts;
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

        auto params = instance.get_impl_params();

        ov::TensorVector input_host_tensors;
        ov::TensorVector output_host_tensors;

        std::vector<memory::ptr> input_mem_ptrs;
        for (size_t i = 0; i < instance.dependencies().size(); i++)
            input_mem_ptrs.push_back(instance.dep_memory_ptr(i));

        auto output_mem_ptr = instance.output_memory_ptr();

        cldnn::mem_lock<uint8_t, mem_lock_type::read_write> output_lock(output_mem_ptr, stream);

        for (size_t i = 0; i < input_mem_ptrs.size(); i++)
            input_host_tensors.push_back(make_tensor(params->input_layouts[i], input_mem_ptrs[i]->lock(stream, mem_lock_type::read)));

        output_host_tensors.push_back(make_tensor(params->output_layouts[0], output_lock.data()));
//        input_host_tensors.push_back(axis_tensor);

//        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
//                        "[GPU] Couldn't execute moe_gather primitive with id ", instance.id());

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
    static std::unique_ptr<primitive_impl> create(const moe_gather_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_gather_impl>();
    }
};


namespace detail {

attach_moe_gather_impl::attach_moe_gather_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
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
