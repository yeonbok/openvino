// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "moe_gemm_gen_micro.hpp"
// clang-format on
#include "moe_gemm.hpp"
#include "moe_gemm_base.hpp"

#include "moe_gemm_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {


class MoEGemmImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoEGemmImpl)

    Stage::Ptr regular_micro_multi_tokens = make_stage<MoEGemmMicroGenerator>(true);

    explicit MoEGemmImpl() : PrimitiveImplOCL(MoEGemm::get_type_info_static()) {}
    explicit MoEGemmImpl(const RuntimeParams& impl_param) : MoEGemmImpl() {
        auto params = impl_param;
        GPU_DEBUG_TRACE_DETAIL << "create stages for dynamic = " << params.is_dynamic() << "\n";
        add_stage(regular_micro_multi_tokens, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoEGemmImpl>(this);
    }
    
    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoEGemmRuntimeParams>();
        }
        auto rtp = static_cast<MoEGemmRuntimeParams*>(m_rt_params.get());
        rtp->num_actual_used_experts = instance.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT).get_shape()[0];
        std::cout << "update_rt_params: " << " num_actual_used_experts: " << rtp->num_actual_used_experts << std::endl;
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        inst.update_shape_info_tensor(impl_params);
        update_rt_params(inst);
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
//        const auto& params = *instance.get_impl_params();
        if (has_stage(regular_micro_multi_tokens)) {
            return execute_stage(events, instance, regular_micro_multi_tokens);
        }

        return nullptr;
    }
};
} // namespace

std::unique_ptr<primitive_impl> MoEGemm::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_gemm>());
    std::cout << __FILE__ << " : " << __LINE__ << std::endl;
    std::cout << "create impl " << std::endl;
    return std::make_unique<MoEGemmImpl>(params);
}
}  // namespace ov::intel_gpu::ocl

//BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
//BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoEGemmImpl)
