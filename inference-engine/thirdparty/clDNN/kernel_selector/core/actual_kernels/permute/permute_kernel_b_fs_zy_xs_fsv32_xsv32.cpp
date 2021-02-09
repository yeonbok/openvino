// Copyright (c) 2016-2020 Intel Corporation
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


#include "permute_kernel_b_fs_zy_xs_fsv32_xsv32.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey PermuteKernel_b_fs_zy_xs_fsv32_xsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants PermuteKernel_b_fs_zy_xs_fsv32_xsv32::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    std::vector<std::string> in_idx;
    std::vector<std::string> out_idx;
    switch (params.inputs[0].GetDims().size()) {
        case 6: in_idx = {"b", "f", "x", "y", "z", "w" }; break;
        case 5: in_idx = {"b", "f", "x", "y", "z" }; break;
        default: in_idx = {"b", "f", "x", "y" }; break;
    }

    assert(params.order.size() == in_idx.size());
    for (auto& o : params.order) {
        out_idx.push_back(in_idx[o]);
    }

    std::string input_order = in_idx[0] + "," + in_idx[1];
    std::string output_order = out_idx[0] + "," + out_idx[1];

    for (size_t i = in_idx.size() - 1; i > 1; i--) {
        input_order += "," + in_idx[i];
        output_order += "," + out_idx[i];
    }

    jit.AddConstant(MakeJitConstant("IN_IDX", "INPUT0_GET_INDEX(" + input_order + ")"));
    jit.AddConstant(MakeJitConstant("OUT_IDX", "OUTPUT_GET_INDEX(" + output_order + ")"));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_H", params.tile_h));
    jit.AddConstant(MakeJitConstant("TILE_SIZE_W", params.tile_w));
    jit.AddConstant(MakeJitConstant("LWS", dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2]));
#if 0
    if (!params.fused_ops.empty()) {
        if (out_idx.size() == 4)
            std::swap(out_idx[2], out_idx[3]);
        else if (out_idx.size() == 5)
            std::swap(out_idx[2], out_idx[4]);
        else if (out_idx.size() == 6) {
            std::swap(out_idx[2], out_idx[5]);
            std::swap(out_idx[3], out_idx[4]);
        }

        FusedOpsConfiguration conf = {"", out_idx, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
#endif
    return jit;
}

CommonDispatchData PermuteKernel_b_fs_zy_xs_fsv32_xsv32::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];
    const auto& tile_w = params.tile_w;
    const auto& tile_h = params.tile_h;
    switch (in.GetLayout()) {
//        case DataLayout::bfyx:
//            dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
//            break;
        case DataLayout::bfzyx:
            dispatchData.gws = {in.X().v / tile_w, in.Y().v * in.Z().v, (in.Feature().v / tile_h) * in.Batch().v};
            break;
//        case DataLayout::bfwzyx:
//            dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
//            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
            break;
    }
//    dispatchData.lws = {8, 1, 1};
//    dispatchData.lws = {64, 1, 1};
    dispatchData.lws = {64, 1, 2};

    return dispatchData;
}

KernelsData PermuteKernel_b_fs_zy_xs_fsv32_xsv32::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::PERMUTE);

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());


    //const auto& in = newParams.inputs[0];
    auto dispatchData = SetDefault(newParams);
#if 0
    kernel.workGroups.global = {in.X().v / newParams.tile_w, in.Y().v * in.Z().v, (in.Feature().v / newParams.tile_h) * in.Batch().v};
    kernel.workGroups.local = {8, 1, 1};
#endif

    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    //kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
   // kernel.arguments = GetArgsDesc(1, false, false, GetFusedPrimitiveInputsCount(params));
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority PermuteKernel_b_fs_zy_xs_fsv32_xsv32::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
