// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

#include <vector>

namespace kernel_selector {

class FullyConnected_vec_mat_tiled : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;
    FullyConnected_vec_mat_tiled();

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params, const optional_params& /*options*/) const override;

protected:
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE };
    }
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
