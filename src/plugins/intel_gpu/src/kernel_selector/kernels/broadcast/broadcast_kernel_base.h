﻿// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_params : public base_params {
    broadcast_params() : base_params(KernelType::BROADCAST) {}
    std::vector<uint16_t> input_order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// broadcast_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct broadcast_optional_params : optional_params {
    broadcast_optional_params() : optional_params(KernelType::BROADCAST) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BroadcastKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BroadcastKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

protected:
    virtual JitConstants GetJitConstants(const broadcast_params& params) const;
    virtual DispatchData SetDefault(const broadcast_params& params) const;
    virtual KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    std::string GetInputBlockND(const broadcast_params& params) const;
};
}  // namespace kernel_selector
