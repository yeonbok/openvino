// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_stream.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "ze_command_list.hpp"
#include "ze_api.h"
#include "ze_event_pool.hpp"
#include "ze_event.hpp"
#include "ze_kernel.hpp"
#include "ze_memory.hpp"
#include "ze_common.hpp"

#include <cassert>
#include <string>
#include <vector>
#include <memory>

namespace cldnn {
namespace ze {


ze_stream::ze_stream(const ze_engine &engine, const ExecutionConfig& config)
    : stream(config.get_property(ov::intel_gpu::queue_type), stream::get_expected_sync_method(config))
    , _engine(engine)
    , m_pool(engine.create_events_pool(100, config.get_property(ov::enable_profiling))) {
    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = 0;
    command_queue_desc.ordinal = 0;
    command_queue_desc.flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_command_list));
    ZE_CHECK(zeCommandQueueCreate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_queue));
}

ze_stream::~ze_stream() {
    zeCommandListDestroy(m_command_list);
    zeCommandQueueDestroy(m_queue);
}

void ze_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);
    auto& kern = ze_kernel.get_handle();
    set_arguments_impl(kern, args_desc.arguments, args);
}

event::ptr ze_stream::enqueue_kernel(kernel& kernel,
                                     const kernel_arguments_desc& args_desc,
                                     const kernel_arguments_data& /* args */,
                                     std::vector<event::ptr> const& deps,
                                     bool is_output) {
    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);

    auto& kern = ze_kernel.get_handle();

    std::vector<ze_event_handle_t> dep_events;
    std::vector<ze_event_handle_t>* dep_events_ptr = nullptr;
    if (m_sync_method == SyncMethods::events) {
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }
    bool set_output_event = m_sync_method == SyncMethods::events || is_output;

    auto ev = set_output_event ? create_base_event() : create_user_event(true);
    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t args = { global.groupCountX / local.groupCountX, global.groupCountY / local.groupCountY, global.groupCountZ / local.groupCountZ };
    ZE_CHECK(zeKernelSetGroupSize(kern, local.groupCountX, local.groupCountY, local.groupCountZ));
    ZE_CHECK(zeCommandListAppendLaunchKernel(m_command_list,
                                             kern,
                                             &args,
                                             set_output_event ? std::dynamic_pointer_cast<ze_base_event>(ev)->get() : nullptr,
                                             dep_events_ptr == nullptr ? 0 : static_cast<uint32_t>(dep_events_ptr->size()),
                                             dep_events_ptr == nullptr ? 0 : &dep_events_ptr->front()));

    return ev;
}

void ze_stream::enqueue_barrier() {
    ZE_CHECK(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    if (deps.empty()) {
        auto ev = create_base_event();
        ZE_CHECK(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_base_event>(ev)->get(), 0, nullptr));
        return ev;
    }

    if (m_sync_method  == SyncMethods::events) {
        std::vector<ze_event_handle_t> dep_events;
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
            }
        }
        if (dep_events.empty())
            return create_user_event(true);

        auto ev = create_base_event();
        ZE_CHECK(zeCommandListAppendBarrier(m_command_list,
                                            std::dynamic_pointer_cast<ze_base_event>(ev)->get(),
                                            static_cast<uint32_t>(dep_events.size()),
                                            &dep_events.front()));
        return ev;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        assert(m_last_barrier_ev != nullptr);
        return m_last_barrier_ev;
    } else {
        return create_user_event(true);
    }
}

ze_event::ptr ze_stream::group_events(std::vector<ze_events::ptr> const& deps) {
    return std::make_shared<ze_events>(deps);
}

void ze_stream::wait() {
    finish();
}

event::ptr ze_stream::create_user_event(bool set) {
    auto ev = m_pool->create_user_event();
    if (set)
        ev->set();

    return ev;
}

event::ptr ze_stream::create_base_event() {
    return m_pool->create_event(++m_queue_counter);
}

void ze_stream::flush() const { }

void ze_stream::finish() const {
    ZE_CHECK(zeCommandListHostSynchronize(m_command_list, default_timeout));
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    for (auto& ev : events) {
        ev->wait();
    }
}

void ze_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(dep.get());
        assert(ze_base_ev != nullptr);
        if (ze_base_ev->get_queue_stamp() > m_last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        if (is_output) {
            m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_base_event());
            m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
            ZE_CHECK(zeCommandListAppendBarrier(m_command_list, m_last_barrier_ev->get(), 0, nullptr));
        } else {
            ZE_CHECK(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
        }
        m_last_barrier = ++m_queue_counter;
    }

    if (!m_last_barrier_ev) {
        m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_user_event(true));
        m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
    }
}

command_list::ptr ze_stream::create_command_list() const {
    return std::make_shared<ze_command_list>(_engine);
}

event::ptr ze_stream::enqueue_command_list(command_list& list) {
    auto ze_list = downcast<ze_command_list>(list).get_handle();
    ZE_CHECK(zeCommandQueueExecuteCommandLists(m_queue, 1, &ze_list, nullptr));
    ZE_CHECK(zeCommandQueueSynchronize(m_queue, -1));

    auto out_ev = list.get_output_event();
    auto ze_ev_handle = downcast<ze_event>(out_ev.get())->get();

    m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_base_event());
    ZE_CHECK(zeCommandListAppendBarrier(m_command_list, m_last_barrier_ev->get(), 1, &ze_ev_handle));
    // ZE_CHECK(zeCommandListImmediateAppendCommandListsExp(m_command_list, 1, &ze_list, nullptr, 0, nullptr));
    return m_last_barrier_ev;
}

}  // namespace ze
}  // namespace cldnn
