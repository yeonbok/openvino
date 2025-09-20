// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "moe_gemm.hpp"

#include "moe_gemm_inst.h"

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {
class MoEGemmGenerator : public KernelGenerator {
public:
    MoEGemmGenerator() : KernelGenerator("moe_gemm") {}
protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
//        auto desc = params.typed_desc<moe_gemm>();

        constexpr size_t subgroup_size = 16;
        jit_constants.make("SUBGROUP_SIZE", subgroup_size);
        return jit_constants;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        args.push_back({ArgumentDescriptor::Types::INPUT, 0}); // input
        args.push_back({ArgumentDescriptor::Types::INPUT, 1}); // weight
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2}); // input offset
        args.push_back({ArgumentDescriptor::Types::INPUT, 3}); // weight offset
        args.push_back({ArgumentDescriptor::Types::INPUT, 4}); // n_array 
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // k
        return args;
    }

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
};

class MoEGemmImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoEGemmImpl)

    Stage::Ptr moe_gemm = make_stage<MoEGemmGenerator>();

    MoEGemmImpl() : PrimitiveImplOCL(MoEGemm::get_type_info_static()) {}
    MoEGemmImpl(const program_node& node, const RuntimeParams& params) : MoEGemmImpl() {
        add_stage(moe_gemm, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoEGemmImpl>(this);
    }
};
} // namespace

std::unique_ptr<primitive_impl> MoEGemm::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_gemm>());
    return std::make_unique<MoEGemmImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

//BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoEGemmImpl)
