// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"

#include "moe_gemm_inst.h"
#include "moe_gemm_gen_opt.hpp"
#include "moe_gemm_base.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "micro_utils.hpp"

class MoEGemmMicroGenerator : public MoEGemmOptGeneratorBase {
public:
    explicit MoEGemmMicroGenerator(bool prefill) : MoEGemmOptGeneratorBase("moe_gemm", prefill ? "_prefill" : "_generate") {}

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override {
        auto base_options = KernelGenerator::get_build_options(params);
        std::string extra_options = " -Dcl_intel_dot_accumulate";
        extra_options += " -Dcl_intel_global_float_atomic";
        extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
        extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";

        return base_options + extra_options;
    }

    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm) const;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto& desc = params.typed_desc<moe_gemm>();

            auto& wgs = kd.params.workGroups;
            auto input_layout = params.get_input_layout();
            auto output_layout = params.get_output_layout();

            wgs.global = {1, 1, 1};
            wgs.local = {1, 1, 1};

            auto& scalars = kd.params.scalars;
            scalars.clear();
            scalars.reserve(1);
            ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
            s_k.v.s32 = 16; // TODO
            scalars.push_back(s_k);
    }};
    }

    static void init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe);
    static std::mutex mtx;
};
#endif
}  // namespace ov::intel_gpu::ocl
