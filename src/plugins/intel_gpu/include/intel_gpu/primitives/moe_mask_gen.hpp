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

    enum MoEMaskGenOutputIdx {
        NUM_ACTUALLY_USED_EXPERTS = 0,
        TOKENS_PER_EXPERT = 1,
        EXPERTS_INFO_START_IDX = 2,
        EXPERTS_ID = 3,
        TOKENS_LENS_PER_EXPERT = 4
    };

    moe_mask_gen() : primitive_base("", {}) {}

    /// @brief Constructs moe_mask_gen primitive.
    ///
    /// @param id                   This primitive id.
    /// @param router_idx           TopK output:1
    /// @param output0 :            Num actually used experts
    /// @param output1 :            tokens_per_expert
    /// @param output2 :            experts_info_start_idx
    /// @param output3 :            experts_id
    /// @param output4 :            tokens_lens_per_expert

    moe_mask_gen(const primitive_id& id,
              const input_info& router_idx,
              const int32_t num_total_experts,
              const int32_t num_experts_per_token)
        : primitive_base(id, {router_idx}, 5),
          num_total_experts(num_total_experts),
          num_experts_per_token(num_experts_per_token) {}

    int32_t num_total_experts = 0;
    int32_t num_experts_per_token = 0;

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
