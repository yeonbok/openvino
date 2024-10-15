// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ze_api.h>

#include <limits>
#include "intel_gpu/runtime/kernel_args.hpp"

#define ZE_CHECK(f) \
    do { \
        ze_result_t res_ = (f); \
        if (res_ != ZE_RESULT_SUCCESS) { \
            throw std::runtime_error(#f " command failed with code " + std::to_string(res_)); \
        } \
    } while (false)


namespace cldnn {
namespace ze {

static constexpr uint64_t default_timeout = std::numeric_limits<uint64_t>::max();

void* find_ze_symbol(const char *symbol);

template <typename F>
F find_ze_symbol(const char *symbol) {
    return (F)find_ze_symbol(symbol);
}

void set_arguments_impl(ze_kernel_handle_t kernel, const arguments_desc& args, const kernel_arguments_data& data);
std::pair<size_t, void*> get_arguments_impl(const argument_desc& desc, const kernel_arguments_data& data);

inline ze_group_count_t to_group_count(const std::vector<size_t>& v) {
     switch (v.size()) {
        case 1:
            return {uint32_t(v[0]), uint32_t(1), uint32_t(1)};
        case 2:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(1)};
        case 3:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(v[2])};
        default:
            return {uint32_t(1), uint32_t(1), uint32_t(1)};
    }
}

}  // namespace ze
}  // namespace cldnn
