// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct moe_gemm : public primitive_base<moe_gemm> {
    CLDNN_DECLARE_PRIMITIVE(moe_gemm)

    enum MoEGemmInputIdx {
        INPUT = 0,
        WEIGHT = 1,
        EXPERTS_IDS = 2,
        INPUT_OFFSET_PER_EXPERT = 3,
        INPUT_TOKENS_LENS = 4
    };

    moe_gemm() : primitive_base("", {}) {}

    /// @brief Constructs moe_gemm primitive.
    ///
    moe_gemm(const primitive_id& id,
             const input_info& input,
             const input_info& weight,
             const input_info& experts_ids,
             const input_info& inputs_offset_per_expert,
             const input_info& input_tokens_lens,
             const int32_t num_active_experts)
        : primitive_base(id, {input, weight, experts_ids, inputs_offset_per_expert, input_tokens_lens}),
          num_active_experts(num_active_experts) {}

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
        primitive_base<moe_gemm>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_gemm>::load(ib);
    }
};
}
