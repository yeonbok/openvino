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

    moe_gemm() : primitive_base("", {}) {}

    /// @brief Constructs moe_gemm primitive.
    ///
    moe_gemm(const primitive_id& id,
              const input_info& input,
              const input_info& weight,
              const input_info& input_offsets,
              const input_info& weight_offsets,
              const input_info& input_tokens_lens)
        : primitive_base(id, {input, weight, input_offsets, weight_offsets, input_tokens_lens}) {}

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
