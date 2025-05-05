// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_nf4.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ReorderWeightsKernelNF4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::NF4);
    k.EnableOutputWeightsType(WeightsType::NF4);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

KernelsData ReorderWeightsKernelNF4::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

ReorderWeightsKernelNF4::DispatchData ReorderWeightsKernelNF4::SetDefault(const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.output;

    // Divide OFM by 2 to save with byte granularity
    dispatchData.gws = { CeilDiv(output.OFM().v, 2), output.IFM().v, 1 };
    dispatchData.lws = { 1, 1, 1 };

    return dispatchData;
}

bool ReorderWeightsKernelNF4::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;

    if (input.LogicalSize() != input.OFM().v * input.IFM().v ||
        output.LogicalSize() != output.OFM().v * output.IFM().v) {
        std::cerr << " Logical size issue!\n";
        return false;
    }

    return true;
}

KernelsPriority ReorderWeightsKernelNF4::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
