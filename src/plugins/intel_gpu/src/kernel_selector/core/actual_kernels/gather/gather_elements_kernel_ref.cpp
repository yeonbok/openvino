// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey GatherElementsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    // k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    // k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    // k.EnableInputLayout(DataLayout::bfwzyx);
    // k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static inline std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = { "b", "f", "y", "x" };
    } else if (size == 5) {
        default_order = { "b", "f", "z", "y", "x" };
    } else if (size == 6) {
        default_order = { "b", "f", "w", "z", "y", "x" };
    }

    return default_order;
}

CommonDispatchData GatherElementsKernelRef::SetDefault(const gather_elements_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    switch (params.inputs[1].GetLayout()) {
    case DataLayout::bfyx:
        //fully parallelizable -> 모든 indices마다 커널할당
        dispatchData.gws = { params.inputs[1].X().v, params.inputs[1].Y().v, params.inputs[1].Feature().v*params.inputs[1].Batch().v };
        break;

    case DataLayout::bfzyx:
        dispatchData.gws = { params.inputs[1].X().v, params.inputs[1].Y().v*params.inputs[1].Z().v, params.inputs[1].Feature().v*params.inputs[1].Batch().v };
        break;

    // case DataLayout::bfwzyx:
    //     dispatchData.gws = { indices_dims[5] * indices_dims[4], indices_dims[3] * indices_dims[2], indices_dims[1] * indices_dims[0] };
    //     break;

    default:
        throw std::invalid_argument("Unsupported data layout for scatter elements update primitive");
        break;
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

JitConstants GatherElementsKernelRef::GetJitConstants(const gather_elements_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    /*
        #if AXIS==0
            #define AXIS_LEN0 INPUT0_BATCH_NUM
            #define AXIS_LEN1 INPUT1_BATCH_NUM
        #elif AXIS==1
            #define AXIS_LEN0 INPUT0_FEATURE_NUM
            #define AXIS_LEN1 INPUT1_FEATURE_NUM
        #elif AXIS==2
            #define AXIS_LEN0 INPUT0_SIZE_Y
            #define AXIS_LEN1 INPUT1_SIZE_Y
        #else
            #define AXIS_LEN0 INPUT0_SIZE_X
            #define AXIS_LEN1 INPUT1_SIZE_X
        #endif
    */

    auto dims = params.inputs[0].LogicalDims();
    std::reverse(dims.begin(), dims.end());
    jit.AddConstant(MakeJitConstant("AXIS", params.axis));
    jit.AddConstant(MakeJitConstant("AXIS_LEN0", dims[params.axis]));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = { "", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

bool GatherElementsKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType:: GATHER_ELEMENTS || o.GetType() != KernelType::GATHER_ELEMENTS) {
        return false;
    }

    const gather_elements_params& params = static_cast<const gather_elements_params&>(p);
    auto input_dims = params.inputs[0].LogicalDims();
    auto indices_dims = params.inputs[1].LogicalDims();
    auto axis = params.axis;

    //왜 뒤집혀있을까..
    std::reverse(input_dims.begin(), input_dims.end());
    std::reverse(indices_dims.begin(), indices_dims.end());

    int input_axisdim = input_dims[axis];
    int indices_axisdim = indices_dims[axis];

    input_dims[axis] = indices_dims[axis]=-1;
    if ( input_dims != indices_dims )
        return false;
    input_dims[axis] = input_axisdim;
    indices_dims[axis] = indices_axisdim;

    int dim = input_dims.size();
    if ( axis < 0 || axis >= dim )
        return false;
    //TODO: indices가 조건 만족하는지 등의 추가조건 검사
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData GatherElementsKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    //여기까지는 옴. 두번째 넷웍 실행(=퓨즈드 온)에서 여기  못오는듯?
    KernelData kd = KernelData::Default<gather_elements_params>(params);
    gather_elements_params& newParams = *static_cast<gather_elements_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams, options);
    auto cldnn_jit = GetJitConstants(newParams);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2, GetFusedPrimitiveInputsCount(params));

    return { kd };
}

}  // namespace kernel_selector
