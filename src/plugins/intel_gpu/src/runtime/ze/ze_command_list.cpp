// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_command_list.hpp"
#include <memory>
#include "intel_gpu/runtime/utils.hpp"
#include "ze/ze_engine.hpp"
#include "ze/ze_kernel.hpp"
#include "ze/ze_memory.hpp"
#include "ze_api.h"

#define MUTABLE 0

namespace cldnn {
namespace ze {

ze_command_list::ze_command_list(const ze_engine& engine)
    : m_engine(engine)
    , m_pool(engine.create_events_pool(2, false)) {
}

void ze_command_list::start() {
    if (m_command_list)
        reset();

#if MUTABLE
    ze_mutable_command_list_exp_desc_t mutable_list_desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0 };
    ze_command_list_desc_t command_list_desc = { ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_list_desc, 0, 0 };
#else
    ze_command_list_desc_t command_list_desc = { ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, 0, 0 };
#endif
    ZE_CHECK(zeCommandListCreate(m_engine.get_context(), m_engine.get_device(), &command_list_desc, &m_command_list));

    ZE_CHECK(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

void ze_command_list::mutate_command(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    ze_group_count_t groupCount = to_group_count(args_desc.workGroups.global);
    auto command_id = downcast<ze_kernel>(k)._cmd_id;

    ze_mutable_group_count_exp_desc_t group_count_desc = { ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC, nullptr, command_id, &groupCount };

    std::vector<ze_mutable_kernel_argument_exp_desc_t> kernel_args_desc;
    for (size_t i = 0; i < args_desc.arguments.size(); i++) {
        void* prev = i > 0 ? static_cast<void*>(&kernel_args_desc[i - 1]) : static_cast<void*>(&group_count_desc);
        auto kernel_arg = get_arguments_impl(args_desc.arguments[i], args);
        ze_mutable_kernel_argument_exp_desc_t kernel_arg_desc{
            ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC,
            prev,
            command_id,
            0,
            kernel_arg.first,
            kernel_arg.second
        };
        kernel_args_desc.push_back(kernel_arg_desc);
    }

    ze_mutable_commands_exp_desc_t desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC, &kernel_args_desc[kernel_args_desc.size() - 1], 0 };

    zeCommandListUpdateMutableCommandsExp(m_command_list, &desc);

    // // Update signal event for the launch kernel command
    // zeCommandListUpdateMutableCommandSignalEventExp(m_command_list, commandId, hNewLaunchKernelSignalEvent);

    // // Update the wait events for the launch kernel command
    // zeCommandListUpdateMutableCommandWaitEventsExp(m_command_list, commandId, 1, &hNewLaunchKernelWaitEvent);
}

void ze_command_list::add(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    auto& casted = downcast<ze_kernel>(k);
    casted._cmd_id = get_command_id();
    auto& ze_handle = casted.get_handle();

    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t ze_args = { global.groupCountX / local.groupCountX, global.groupCountY / local.groupCountY, global.groupCountZ / local.groupCountZ };

    set_arguments_impl(ze_handle, args_desc.arguments, args);
    ZE_CHECK(zeKernelSetGroupSize(ze_handle, local.groupCountX, local.groupCountY, local.groupCountZ));
    ZE_CHECK(zeCommandListAppendLaunchKernel(m_command_list, ze_handle, &ze_args, nullptr, 0, nullptr));
}

void ze_command_list::close() {
    m_output_event = m_pool->create_event();
    ZE_CHECK(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_event>(m_output_event)->get(), 0, nullptr));
    ZE_CHECK(zeCommandListClose(m_command_list));
}

void ze_command_list::reset() {
    ZE_CHECK(zeCommandListDestroy(m_command_list));
}

ze_command_list::~ze_command_list() {
    reset();
}

uint64_t ze_command_list::get_command_id() {
    if (is_mutable()) {
        ze_mutable_command_exp_flags_t flags =
            ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE;

        ze_mutable_command_id_exp_desc_t cmd_id_desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC, nullptr, flags };
        uint64_t cmd_id = 0;
        ZE_CHECK(zeCommandListGetNextCommandIdExp(m_command_list, &cmd_id_desc, &cmd_id));
        return cmd_id;
    } else {
        thread_local uint64_t cmd_id = 0;
        cmd_id++;

        return cmd_id;
    }
}

event::ptr ze_command_list::get_output_event() const {
    assert(m_output_event != nullptr);
    return m_output_event;
}

}  // namespace ze
}  // namespace cldnn
