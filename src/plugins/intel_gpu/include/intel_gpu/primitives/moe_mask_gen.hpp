// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_mask_gen : public primitive_base<moe_mask_gen> {
    CLDNN_DECLARE_PRIMITIVE(moe_mask_gen)

    moe_mask_gen() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen primitive.
    ///
    /// @param id                   This primitive id.
    /// @param input                Input data primitive id.
    /// @param experts_map          experts map per input token
//    moe_mask_gen(const primitive_id& id,
//              const input_info& router_idx,
//              const int32_t num_total_experts,
//              const int32_t num_active_experts)
//        : primitive_base(id, {router_idx}, 2),
//          num_total_experts(num_total_experts),
//          num_active_experts(num_active_experts) {}


    moe_mask_gen(const primitive_id& id,
              const input_info& router_idx,
              const input_info& router_weight,
              const int32_t num_total_experts,
              const int32_t num_active_experts)
//        : primitive_base(id, {router_idx, router_weight}, 2),
        : primitive_base(id, {router_idx, router_weight}),
          num_total_experts(num_total_experts),
          num_active_experts(num_active_experts) {}

    int32_t num_total_experts = 0;
    int32_t num_active_experts = 0;

    size_t hash() const override {
        return primitive::hash();
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<moe_mask_gen>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_mask_gen>::load(ib);
    }
};
}
