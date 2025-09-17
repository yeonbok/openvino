// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_gather : public primitive_base<moe_gather> {
    CLDNN_DECLARE_PRIMITIVE(moe_gather)

    moe_gather() : primitive_base("", {}) {}

    /// @brief Constructs moe_gather primitive.
    ///
    /// @param id                   This primitive id.
    /// @param input                Input data primitive id.
    /// @param experts_map          experts map per input token
    moe_gather(const primitive_id& id,
              const input_info& data,
              const input_info& tokens_per_expert,
              const input_info& expert_offset,
              const int num_active_experts)
        : primitive_base(id, {data, tokens_per_expert, expert_offset}),
                         num_active_experts(num_active_experts) {}

    int num_active_experts = 0;

    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto rhs_casted = downcast<const moe_gather>(rhs);
        return num_active_experts == rhs_casted.num_active_experts;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_gather>::save(ob);
        ob << num_active_experts;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gather>::load(ib);
        ib >> num_active_experts;
    }
};
}
