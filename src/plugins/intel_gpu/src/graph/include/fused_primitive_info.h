// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "kernel_selector_params.h"
#include "meta_utils.h"

namespace cldnn {
struct program_node;

using primitive_type_id = struct primitive_type *;

//template <class PType>
struct fused_primitive_desc {
//    std::shared_ptr<program_node>& node;

    template <class PType>
    bool is_type() const {
        static_assert(
            meta::is_primitive<PType>::value,
            "Type argument for program_node::is_type should be a non-const, non-volatile type derived from primitive");
        return desc->type == PType::type_id();
    }
    template <class PType>
    std::shared_ptr<const PType> typed_desc() const { return std::static_pointer_cast<const PType>(desc); }
    template<typename T>
    std::shared_ptr<T> get_type_params() const {
        auto p = std::dynamic_pointer_cast<T>(f_param);
        if (!p)
            throw std::runtime_error("Invalid dynamic cast of fused parameters!");
        return p;
    }

    std::shared_ptr<const primitive> desc;
//    const primitive_type_id type;
    std::shared_ptr<kernel_selector::fuse_params> f_param;
    size_t dep_start_idx;
    std::map<primitive_id, size_t> deps;
    std::map<primitive_id, size_t> fused_deps;
    size_t total_num_deps = 0;
    activation_func activation;
    activation_additional_params activation_params;
    layout input_layout = layout(data_types::f32, format::bfyx, tensor());
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());
    fused_primitive_desc(std::shared_ptr<const primitive> prim) : desc(prim) {}
};
} // namespace cldnn
