﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants GRNKernelBase::GetJitConstants(const grn_params& params, GRNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("BIAS", params.bias));

    return jit;
}

GRNKernelBase::DispatchData GRNKernelBase::SetDefault(const grn_params& params) const {
    const auto& output = params.output;

    DispatchData dispatchData;
    dispatchData.gws = { output.Batch().v, output.Y().v, output.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData GRNKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options) const {
    assert(params.GetType() == KernelType::GRN);

    if (!Validate(params, options))
        return {};

    const grn_params& orgParams = static_cast<const grn_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<grn_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    return {kd};
}

}  // namespace kernel_selector
