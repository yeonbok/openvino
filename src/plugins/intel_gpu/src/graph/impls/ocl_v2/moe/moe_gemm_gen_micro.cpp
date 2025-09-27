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

static const int subgroup_size = 8;

JitConstants MoEGemmMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm) const {
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", subgroup_size);
    constexpr static std::array input_ids = { moe_gemm::MoEGemmInputIdx::INPUT,
                                              moe_gemm::MoEGemmInputIdx::WEIGHT,
                                              moe_gemm::MoEGemmInputIdx::INPUT_OFFSETS,
                                              moe_gemm::MoEGemmInputIdx::WEIGHT_OFFSETS,
                                              moe_gemm::MoEGemmInputIdx::INPUT_TOKENS_LENS
                                            };
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;

    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }

    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
    // TODO
    return jit;
}

std::mutex MoEGemmMicroGenerator::mtx;
void MoEGemmMicroGenerator::init_microkernels(const kernel_impl_params& params,
                                           micro::Package& gemm_moe) {
    // TODO: Remove once micro API is thread safe
    std::lock_guard<std::mutex> l(mtx);

    const auto& device_info = params.get_device_info();
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;

//    const auto key_cache_id = 4; // TODO

    // TODO
//    size_t batch = params.get_input_layout(1).get_shape()[0];
    size_t batch = 1;
//    size_t m = params.get_input_layout(1).get_shape()[1];
//    size_t k = params.get_input_layout(1).get_shape()[2];
    size_t m = 16;
    size_t n = 10;
    size_t k = 16;
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

    /* Set up microkernel options */
//    micro::GEMMProtocol::Options opts_moe;
//    opts_moe.localB = true;
//    opts_moe.slmPtr = true;

    /* Set up problem_moe size information */
    micro::SizeParams sizes;
    // TODO fix
    sizes.n = n; 
    sizes.m = m;
    sizes.k = k;
    sizes.batch = batch;

    /* Set up microkernel requirements */
//    int unroll_m = 4;
//    int unroll_n = 4;
//    int wg_m = 2;
//    int wg_n = 2;
//    std::vector<micro::StrategyRequirement> reqs_moe;
//    reqs_moe.push_back(micro::StrategyRequirement::UnrollM == unroll_m);
//    reqs_moe.push_back(micro::StrategyRequirement::UnrollN == unroll_n);
//    reqs_moe.push_back(micro::StrategyRequirement::WGM == wg_m);
//    reqs_moe.push_back(micro::StrategyRequirement::WGN == wg_n);

    /* Ask microkernel provider for microkernel */
    try {
 //       gemm_moe = micro::select_gemm_microkernel(micro::GEMMProtocol{}, hw_info, sizes, problem_moe, reqs_moe);
        gemm_moe = micro::select_gemm_microkernel(micro::GEMMProtocol{}, hw_info, sizes, problem_moe);
    } catch (const std::runtime_error& ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create moe micro kernel: " << ex.what() << "\n";
        std::cout << "Can't create moe micro kernel: " << ex.what() << "\n";
        throw;
    }
}
DispatchDataFunc MoEGemmMicroGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        const auto& desc = params.typed_desc<moe_gemm>();
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_tile_m = gemm_p.getSetting("sg_tile_m");
        auto sg_tile_n = gemm_p.getSetting("sg_tile_n");

        auto& wgs = kd.params.workGroups;
        auto input_layout = params.get_input_layout(0);
        auto experts_weight_layout = params.get_input_layout(1);
        auto input_offset_layout = params.get_input_layout(2);
        auto weight_offset_layout = params.get_input_layout(3);
        auto input_tokens_lens_layout = params.get_input_layout(4);
        auto output_layout = params.get_output_layout();

        size_t num_active_experts = input_offset_layout.get_shape()[0];
        // input : [num_experts, n, k]
        // experts_weight : [num_experts, m, k]
        size_t M = experts_weight_layout.get_shape()[1];
        size_t N = input_layout.get_shape()[1];
        wgs.local = { sg_per_wg_m * subgroup_size,
                      sg_per_wg_n,
                      1};
        wgs.global = { align_to(ceil_div(M, sg_tile_m), sg_per_wg_m) * subgroup_size,
                       align_to(ceil_div(N, sg_tile_n), sg_per_wg_n),
                       num_active_experts};
        std::cout << "output layout : " << output_layout.to_short_string() << std::endl;
        std::cout << "gws : " << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << std::endl;
    }};
}

std::string MoEGemmMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
//    std::string extra_options = " -Dcl_intel_dot_accumulate";
//    extra_options += " -Dcl_intel_global_float_atomic";
    std::string extra_options = " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    return base_options + extra_options;
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
    args.push_back({ArgumentDescriptor::Types::INPUT, 4});   // out offset // TODO
    args.push_back({ArgumentDescriptor::Types::INPUT, 5});   // n_array
    return args;
}

KernelData MoEGemmMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    std::cout << "get kernel data for micro " << get_kernel_name() << std::endl;
    micro::Package moe_gemm;
    init_microkernels(params, moe_gemm); // TODO

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
    shim_options.subgroupSize = static_cast<int32_t>(subgroup_size);
    shim_options.useTileOps = true;
    shim_options.decorator = "moe";

    kd.code->jit += generateShim(moe_gemm, micro::HostLanguage::OpenCL_C, shim_options);
//    if (moe_gemm.grfMin > 128) {
    kd.code->options += " -cl-intel-256-GRF-per-thread";
//    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(moe_gemm));
    return kd;
}
}  // namespace ov::intel_gpu::ocl
#endif
