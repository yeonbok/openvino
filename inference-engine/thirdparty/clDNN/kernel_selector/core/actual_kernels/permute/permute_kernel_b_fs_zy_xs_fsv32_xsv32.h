// Copyright (c) 2016-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once

#include "kernel_base_opencl.h"
#include "permute_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernel_b_fs_zy_xs_fsv32_xsv32
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PermuteKernel_b_fs_zy_xs_fsv32_xsv32 : public KernelBaseOpenCL {
public:
    PermuteKernel_b_fs_zy_xs_fsv32_xsv32() : KernelBaseOpenCL("permute_b_fs_zy_xs_fsv32_xsv32") {}
    virtual ~PermuteKernel_b_fs_zy_xs_fsv32_xsv32() {}

    virtual CommonDispatchData SetDefault(const permute_params& params) const;
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
