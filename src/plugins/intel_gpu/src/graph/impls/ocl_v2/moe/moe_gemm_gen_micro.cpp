// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "moe_gemm_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "moe_gemm_inst.h"
#include "../utils/kernel_generator.hpp"

// clang-format on
namespace ov::intel_gpu::ocl {

std::mutex MoEGemmMicroGenerator::mtx;

JitConstants MoEGemmMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm) const {
    auto jit = make_base_jit_constants(params);
    constexpr size_t subgroup_size = 16;
    jit.make("SUBGROUP_SIZE", subgroup_size);
    // TODO
    return jit;
}

void MoEGemmMicroGenerator::init_microkernels(const kernel_impl_params& params,
                                           micro::Package& gemm_moe) {
    // TODO: Remove once micro API is thread safe
    const auto& device_info = params.get_device_info();
    std::lock_guard<std::mutex> l(mtx);
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;

//    const auto key_cache_id = 4; // TODO

    // TODO
    int k = 16; 
    micro::GEMMProblem problem_moe;
    problem_moe.Ta = problem_moe.Ta_ext = micro::Type::f16;
    problem_moe.Tb = problem_moe.Tb_ext = micro::Type::f16;
    problem_moe.Tc = problem_moe.Tc_ext = micro::Type::f32;
    problem_moe.Ts = problem_moe.Tc;
    problem_moe.A.layout = micro::MatrixLayout::T;
    problem_moe.B.layout = micro::MatrixLayout::N;
    problem_moe.C.layout = micro::MatrixLayout::N;
    problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta));
    problem_moe.B.setAlignment(micro::alignment_for_ld(k * problem_moe.Tb));
    problem_moe.C.setAlignment(problem_moe.Tc.size());

    /* Set up problem_moe size information */
    micro::SizeParams sizes;
    // TODO fix
    sizes.m = 30;
    sizes.n = 10; 
    sizes.k = 16;
    sizes.batch = 1;

    /* Set up microkernel requirements */
    std::vector<micro::StrategyRequirement> reqs_moe;
//    reqs_moe.push_back(micro::StrategyRequirement::UnrollM == config->unroll_m_moe);
//    reqs_moe.push_back(micro::StrategyRequirement::UnrollN == config->unroll_n_moe);
//    reqs_moe.push_back(micro::StrategyRequirement::WGM == config->wg_m_moe);
//    reqs_moe.push_back(micro::StrategyRequirement::WGN == config->wg_n_moe);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_moe = micro::select_gemm_microkernel(micro::GEMMProtocol{}, hw_info, sizes, problem_moe, reqs_moe);
    } catch (const std::runtime_error& ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create moe sdpa_micro kernel: " << ex.what() << "\n";
        throw;
    }
}

Arguments MoEGemmMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // input
    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // weight
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});   // input offset
    args.push_back({ArgumentDescriptor::Types::INPUT, 3});   // weight offset
    args.push_back({ArgumentDescriptor::Types::INPUT, 4});   // n_array
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // k
    return args;
}

KernelData MoEGemmMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    std::cout << "get kernel data for micro" << std::endl;
    micro::Package moe_gemm;
    init_microkernels(params, moe_gemm); // TODO

//    const auto& device_info = params.get_device_info();
    auto jit = get_jit_constants(params, moe_gemm);

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = kernel_language::OCLC_V2;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = false;
    kd.code->has_microkernels = true;
    kd.code->str = build_code(get_kernel_name(), jit, kd.code->entry_point);

    kd.params.arguments = get_arguments_desc(params);
    kd.update_dispatch_data_func = get_dispatch_data_func();

    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    /* Generate microkernel shims */
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = static_cast<int32_t>(16);
    shim_options.useTileOps = true;
    shim_options.decorator = "moe";

    kd.code->jit += generateShim(moe_gemm, micro::HostLanguage::OpenCL_C, shim_options);
    if (moe_gemm.grfMin > 128) {
        kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(moe_gemm));
    return kd;
}
}  // namespace ov::intel_gpu::ocl
#endif
