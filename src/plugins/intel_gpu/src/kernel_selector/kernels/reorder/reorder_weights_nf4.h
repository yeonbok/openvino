// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderWeightsKernelNF4 : public ReorderKernelBase {
public:
    ReorderWeightsKernelNF4() : ReorderKernelBase("reorder_weights_nf4") {}
    virtual ~ReorderWeightsKernelNF4() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const reorder_weights_params& arg) const override;

protected:
    bool Validate(const Params& params) const override;
};
}  // namespace kernel_selector
