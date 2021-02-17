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


#include "permute_kernel_tile_8x8.h"
#include "kernel_selector_utils.h"
#include <string>
#include <functional>
#include <cmath>

#define CEIL_DIV(A, B) ((A + B - 1)/(B))
#define TILE_SIZE 8 // 8x8

namespace kernel_selector {
ParamsKey PermuteKernel_tile_8x8::GetSupportedKey() const {
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

static inline std::string GetTiledOutputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, y, (x * TILE_SIZE + lh), (TILE_SIZE * f + lw*VECTORWIDTH)";
            break;
        case 5 :
            order_str = "b, z, y, (x * TILE_SIZE + lh), (TILE_SIZE * f + lw*VECTORWIDTH)";
            break;
        case 6 :
            order_str = "b, w, z, y, (x * TILE_SIZE + lh), (TILE_SIZE * f + lw*VECTORWIDTH)";
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;

}

static inline std::string GetTiledInputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, (TILE_SIZE * f + lh), y, (TILE_SIZE * x + lw * VECTORWIDTH)";
            break;
        case 5 :
            order_str = "b, (TILE_SIZE * f + lh), z, y, (TILE_SIZE * x + lw * VECTORWIDTH)";
            break;
        case 6 :
            order_str = "b, (TILE_SIZE * f + lh), w, z, y, (TILE_SIZE * x + lw * VECTORWIDTH)";
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

JitConstants PermuteKernel_tile_8x8::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
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

    int32_t vector_width = TILE_SIZE; // to calculate
    uint64_t total_lws = dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2];
    jit.AddConstant(MakeJitConstant("VECTORWIDTH", vector_width));
    jit.AddConstant(MakeJitConstant("INPUT0_TILED_ORDER", GetTiledInputOrder(params.inputs[0].GetDims().size())));
    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(params.output.GetDims().size())));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", TILE_SIZE));
    jit.AddConstant(MakeJitConstant("N_VECTORS_IN_TILE", TILE_SIZE / vector_width));
    jit.AddConstant(MakeJitConstant("LWS",total_lws));
    jit.AddConstant(MakeJitConstant("NFEATURE_TILES", CEIL_DIV(params.inputs[0].Feature().v, TILE_SIZE)));

    std::string normal_tile_cond = "true";
    std::string x_remainder_cond = "true";
    std::string f_remainder_cond = "true";

    if (params.inputs[0].X().v % TILE_SIZE) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_ITEM", params.inputs[0].X().v / TILE_SIZE));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", params.inputs[0].X().v % TILE_SIZE));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE_AS_VECTOR", CEIL_DIV(params.inputs[0].X().v % TILE_SIZE, vector_width)));
        normal_tile_cond += " && (x < X_REMAINDER_ITEM)";
        x_remainder_cond += " && (x == X_REMAINDER_ITEM)";
        f_remainder_cond += " && (x < X_REMAINDER_ITEM)";
    }
    if (params.inputs[0].Feature().v % TILE_SIZE) {
        jit.AddConstant(MakeJitConstant("F_REMAINDER_ITEM", params.inputs[0].Feature().v / TILE_SIZE));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", params.inputs[0].Feature().v % TILE_SIZE));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE_AS_VECTOR", CEIL_DIV(params.inputs[0].Feature().v % TILE_SIZE, vector_width)));
        normal_tile_cond += " && (f < F_REMAINDER_ITEM)";
        x_remainder_cond += " && (f < F_REMAINDER_ITEM)";
        f_remainder_cond += " && (f == F_REMAINDER_ITEM)";
    }

    jit.AddConstant(MakeJitConstant("NORMAL_TILE_CONDITION", normal_tile_cond));
    jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", x_remainder_cond));
    jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", f_remainder_cond));
    jit.AddConstant(MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, VECTORWIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, VECTORWIDTH)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, VECTORWIDTH)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, VECTORWIDTH)"));
    jit.AddConstant(MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("LOCAL_BUF_STRIDE", (TILE_SIZE / vector_width) * TILE_SIZE));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", (TILE_SIZE / vector_width) * TILE_SIZE * total_lws) );

    if (!params.fused_ops.empty()) {
        if (out_idx.size() == 4)
            std::swap(out_idx[2], out_idx[3]);
        else if (out_idx.size() == 5)
            std::swap(out_idx[2], out_idx[4]);
        else if (out_idx.size() == 6) {
            std::swap(out_idx[2], out_idx[5]);
            std::swap(out_idx[3], out_idx[4]);
        }
        FusedOpsConfiguration conf = {"", out_idx, "input_var", params.inputs[1].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

static std::vector<size_t> GetBestLwsFromGws(const permute_params& params, const std::vector<size_t>& gws)
{
    std::vector<size_t> lws{1, 1, 1};
    std::vector<size_t> dims{0, 2, 1};

    // SLM size: elemsize * TILE_SIZE * TILE_SIZE * work_items <= 64K
    size_t elem_size = sizeof(params.output.GetDType());
    size_t max_local_mem_size = 64 * 1024;
    size_t max_num_work_items = std::min(256lu, max_local_mem_size / (elem_size * TILE_SIZE * TILE_SIZE));

    for (size_t i = 0; i < dims.size(); ++i) {
        size_t dim = dims[i];
        size_t max_divider = static_cast<size_t>(std::sqrt(gws[dim]) + 1);
        for (size_t divider = 1; divider <= max_divider; ++divider)
        {
            if (gws[dim] % divider == 0)
            {
                const size_t lws0 = gws[dim] / divider;
                if (lws0 <= max_num_work_items)
                {
                    lws[dim] = std::max(lws[dim], lws0);
                }
                if (divider <= max_num_work_items)
                {
                    lws[dim] = std::max(lws[dim], divider);
                }
            }
        }
        max_num_work_items /= lws[dim];
    }
    return lws;
}

CommonDispatchData PermuteKernel_tile_8x8::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];
    switch (in.GetLayout()) {
        case DataLayout::bfyx:
            dispatchData.gws = {CEIL_DIV(in.X().v , TILE_SIZE), in.Y().v, CEIL_DIV(in.Feature().v, TILE_SIZE) * in.Batch().v};
            break;
        case DataLayout::bfzyx:
            dispatchData.gws = {CEIL_DIV(in.X().v , TILE_SIZE), in.Y().v * in.Z().v, CEIL_DIV(in.Feature().v, TILE_SIZE) * in.Batch().v};
            break;
        case DataLayout::bfwzyx:
            dispatchData.gws = {CEIL_DIV(in.X().v , TILE_SIZE), in.Y().v * in.Z().v * in.W().v, CEIL_DIV(in.Feature().v, TILE_SIZE) * in.Batch().v};
            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
            break;
    }
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws);
    return dispatchData;
}

KernelsData PermuteKernel_tile_8x8::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::PERMUTE);

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());
    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority PermuteKernel_tile_8x8::GetKernelsPriority(const Params& params/*params*/, const optional_params& /*options*/) const {

    std::function<bool(const std::vector<uint16_t>&)> is_rotating_except_batch = [](const std::vector<uint16_t>& order) {
        // Target transform: Rotate feature dim to back to be taken as inner-most axis
        // ex) 0(b), 4(f), 1(z), 2(y), 3(x)
        if ((int32_t) order[1] != order.size() - 1) return false;
        for (int32_t i = 3; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i] !=  (i - 1)) return false;
        }
        return true;
    };

//    return DONT_USE_IF_HAVE_SOMETHING_ELSE*10;
    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    if (is_rotating_except_batch(newParams.order) && (newParams.inputs[0].Feature().v > TILE_SIZE) && (newParams.inputs[0].X().v > TILE_SIZE)) {
        return FORCE_PRIORITY_1;
    } else
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
