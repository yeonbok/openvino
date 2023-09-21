// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_vec_mat_tiled.h"

#include <vector>
#include <functional>


namespace kernel_selector {

FullyConnected_vec_mat_tiled::FullyConnected_vec_mat_tiled() : FullyConnectedKernelBase("fully_connected_gpu_vec_mat_tiled") {
}

ParamsKey FullyConnected_vec_mat_tiled::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

DeviceFeaturesKey FullyConnected_vec_mat_tiled::get_required_device_features_key(const Params& params, const optional_params& options) const {
    auto k = get_common_subgroups_device_features_key(params, options);
    k.requires_subgroup_shuffle();

    return k;
}

bool FullyConnected_vec_mat_tiled::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];
    auto& output = fc_params.outputs[0];

    // Block reads must be aligned to 4 bytes, for fp16 we can correct for offset misalignment,
    // but we need to ensure that batch pitch preserves alignment.
    if (input.GetDType() == Datatype::F16) {
        if (input.Batch().pitch % 2 != 0 && (input.Batch().v > 1 || fc_params.is_shape_agnostic))
            return false;
        // for 3d case we have to check feature alignment as well
        if (output.GetLayout() == DataLayout::bfyx && input.Feature().pitch % 2 != 0 && (input.Feature().v > 1 || fc_params.is_shape_agnostic))
            return false;
    }

    // Dynamic kernel doesn't support dynamic weights yet
    if (fc_params.is_shape_agnostic && input.is_dynamic()) {
        if ((output.GetLayout() == DataLayout::bfyx && input.Y().v == 0) ||
            (output.GetLayout() == DataLayout::bf && input.Feature().v == 0))
            return false;
    }

    if (input.GetLayout() == DataLayout::bfyx) {
        // Padding on input is not supported.
        // TODO: Enable by mirroring the padding in weights.
        if (input.X().pad.Total() != 0)
            return false;
        if (input.Y().pad.Total() != 0)
            return false;
    }

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx) {
        if (input.X().v > 1)
            return false;
    }

    return true;
}

FullyConnected_vec_mat_tiled::DispatchData
FullyConnected_vec_mat_tiled::SetDefault(const fully_connected_params& params, int autoTuneIndex) const {
    auto dispatchData = Parent::SetDefault(params);
    size_t m_size = params.outputs[0].Batch().v;
    size_t n_size = params.outputs[0].Feature().v;
    size_t n_tile_size = 16;
    size_t m_tile_size = 1; //TODO
    size_t num_n_tiles = CeilDiv(n_size, n_tile_size);
    size_t num_m_tiles = CeilDiv(m_size, m_tile_size);
    dispatchData.gws[0] = num_n_tiles;
    dispatchData.gws[1] = num_m_tiles;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = std::min(params.engineInfo.maxWorkGroupSize, num_n_tiles);
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;
    std::cout << "GWS: [" << num_n_tiles << ", " << 1 << ", " << 1 << "]" << std::endl;
    return dispatchData;
}

KernelsPriority FullyConnected_vec_mat_tiled::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_4; // tmp
}

JitConstants FullyConnected_vec_mat_tiled::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    size_t n_tile_size = 16;
    size_t k_tile_size = 16;
    jit.AddConstant(MakeJitConstant("TILE_N_SIZE", n_tile_size));
    jit.AddConstant(MakeJitConstant("TILE_K_SIZE", k_tile_size));
    jit.AddConstant(MakeJitConstant("VSIZE", 8));
    jit.AddConstant(MakeJitConstant("INPUT_VTYPE", "CAT(INPUT0_TYPE, VSIZE)"));
    jit.AddConstant(MakeJitConstant("FILTER_VTYPE", "CAT(FILTER_TYPE, VSIZE)"));
    jit.AddConstant(MakeJitConstant("OUT_VTYPE", "CAT(OUTPUT_TYPE, VSIZE)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, VSIZE)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, VSIZE)"));
    return jit;
}

KernelsData FullyConnected_vec_mat_tiled::GetKernelsData(const Params& params, const optional_params& optParams) const {
    KernelsData res = {};
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    KernelsData kds = GetCommonKernelsData(params, optParams, fc_params.inputs[0].GetLayout(), WeightsLayout::oiyx);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}

}  // namespace kernel_selector
