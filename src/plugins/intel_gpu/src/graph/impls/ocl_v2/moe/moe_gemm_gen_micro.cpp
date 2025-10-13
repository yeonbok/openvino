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

static size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    case gpu_arch::xe_hpc:
    case gpu_arch::xe2:
    case gpu_arch::xe3:
        return 16;
    default:
        return 0;
    }
}
JitConstants MoEGemmMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));
    std::vector<moe_gemm::MoEGemmInputIdx> input_ids = {moe_gemm::MoEGemmInputIdx::INPUT,
                                                        moe_gemm::MoEGemmInputIdx::WEIGHT,
                                                        moe_gemm::MoEGemmInputIdx::EXPERTS_IDS,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_TOKENS_LENS};
    bool has_bias = false; // TODO
    if (params.input_layouts[1].data_type == data_types::u4 ||
        params.input_layouts[1].data_type == data_types::i4) {
        if (has_bias) {
            input_ids.push_back(moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE);
            input_ids.push_back(moe_gemm::MoEGemmInputIdx::WEIGHT_ZP);
        } else {
            input_ids.push_back((moe_gemm::MoEGemmInputIdx)((int)moe_gemm::MoEGemmInputIdx::WEIGHT_SCALE - 1));
            input_ids.push_back((moe_gemm::MoEGemmInputIdx)((int)moe_gemm::MoEGemmInputIdx::WEIGHT_ZP - 1));
        }
        std::cout << "# is u4||i4" << std::endl;
        jit.make("EXPERT_STRIDE", (params.input_layouts[1].get_shape()[1] * params.input_layouts[1].get_shape()[2]) / 2);
    } else {
        jit.make("EXPERT_STRIDE", params.input_layouts[1].get_shape()[1] * params.input_layouts[1].get_shape()[2]);
    }

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
    jit.make("INPUT_STRIDE", params.input_layouts[1].get_shape()[2]);
    jit.make("OUTPUT_STRIDE", params.input_layouts[1].get_shape()[1]);
    if (!m_is_prefill)
        jit.make("IS_GENERATE", 1);
    // TODO
    if (params.input_layouts[1].data_type == data_types::u4 ||
        params.input_layouts[1].data_type == data_types::i4) {
        std::cout << "weight is compressed int4" << std::endl;
        jit.make("WEIGHT_COMPRESSED_INT4", 1);
        jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
        jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::f16));
    }
    return jit;
}

//static micro::Type convert_type(ov::element::Type t) {
//    switch (t) {
//    case ov::element::f32:
//        return micro::Type::f32;
//    case ov::element::f16:
//        return micro::Type::f16;
//    case ov::element::i8:
//        return micro::Type::s8;
//    case ov::element::u8:
//        return micro::Type::u8;
//    case ov::element::i32:
//        return micro::Type::s32;
//    case ov::element::u4:
//        return micro::Type::u4;
//    default:
//        break;
//    }
//    OPENVINO_THROW("Unsupported element type: ", t);
//}

std::mutex MoEGemmMicroGenerator::mtx;
void MoEGemmMicroGenerator::init_microkernels(const kernel_impl_params& params,
                                           micro::Package& gemm_moe, bool is_prefill) {
    // TODO: Remove once micro API is thread safe
    std::lock_guard<std::mutex> l(mtx);
    std::cout << "init_micro_kernels. is_prefill? " << is_prefill << std::endl;
    const auto& device_info = params.get_device_info();
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;
    bool weight_compressed_u4 = false;
    if (params.input_layouts[1].data_type == data_types::u4 ||
        params.input_layouts[1].data_type == data_types::i4) {
            weight_compressed_u4 = true;
    }
    // TODO
    size_t m = params.get_input_layout(1).get_shape()[1];
    size_t k = params.get_input_layout(1).get_shape()[2];
    std::cout << "init_microkernels (weight u4? " << weight_compressed_u4 << " )" << std::endl;
    size_t n = is_prefill ? 128 : 8; // TODO
    std::cout << "n :" << n << " m : " << m << " k : " << k << std::endl;
    micro::GEMMProblem problem_moe;
    if (weight_compressed_u4) {
        problem_moe.Ta = micro::Type::f16; // weight register
        problem_moe.Ta_ext = micro::Type::u4; // weight memory
        problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta_ext));

        problem_moe.Ta_scale = micro::Type::f16; // scale dtype
        problem_moe.A_scale.setAlignment(2); // scale : half
        problem_moe.A_scale.layout = micro::MatrixLayout::T; // scale layout
        problem_moe.asPtrDims = 2; // A/B scale dimensionality (-1: none; 0: scalar; 1: vector, 2: matrix)

        problem_moe.aqGroupM = 1;
        problem_moe.aqGroupK = k;

        problem_moe.Tao = micro::Type::f16; // zp dt
        problem_moe.AO.setAlignment(2); // zp : half
        problem_moe.AO.layout = micro::MatrixLayout::T;
        problem_moe.aoPtrDims = 2; // // A/B offset dimensionality (-1: none; 0: scalar; 1: vector, 2: matrix)
        problem_moe.aOffset = micro::ABOffset::Calc; // Calculate A/B row/column sums in kernel.
    } else {
        problem_moe.Ta = problem_moe.Ta_ext = micro::Type::f16;
        problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta));
    }

    problem_moe.Tb = problem_moe.Tb_ext = micro::Type::f16;
    problem_moe.Tc = micro::Type::f32;
    problem_moe.Tc_ext = micro::Type::f32;
    problem_moe.Ts = problem_moe.Tc;
    problem_moe.A.layout = micro::MatrixLayout::T;
    problem_moe.B.layout = micro::MatrixLayout::N;
    problem_moe.C.layout = micro::MatrixLayout::N;
    problem_moe.B.setAlignment(micro::alignment_for_ld(k * problem_moe.Tb));
    problem_moe.C.setAlignment(problem_moe.Tc.size());

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_moe;
    opts_moe.slmPtr = true;

    if (weight_compressed_u4) {
        opts_moe.scaleA = opts_moe.offsetA = true;
    }
    /* Set up problem_moe size information */
    micro::SizeParams sizes;
    // TODO fix
    sizes.n = n;
    sizes.m = m;
    sizes.k = k;
    sizes.batch = 1;

    /* Set up microkernel requirements */
//    int unroll_n = is_prefill ? 8 : 1;
//    int unroll_n = is_prefill ? 8 : n;
//    std::vector<micro::StrategyRequirement> reqs_moe;
//    reqs_moe.push_back(micro::StrategyRequirement::UnrollN == unroll_n);
    std::cout << "problem_moe : " << problem_moe.toString() << std::endl;
    /* Ask microkernel provider for microkernel */
    try {
      gemm_moe = micro::select_gemm_microkernel(opts_moe, hw_info, sizes, problem_moe);
    } catch (const std::runtime_error& ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create moe micro kernel: " << ex.what() << "\n";
        std::cout << "Can't create moe micro kernel: " << ex.what() << "\n";
        throw;
    }
    std::cout << "Could create moe micro kernel " << std::endl;
}
DispatchDataFunc MoEGemmMicroGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<MoEGemmRuntimeParams*>(rt_params);
        const auto& desc = params.typed_desc<moe_gemm>();
        const auto& device_info = params.get_device_info();
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_tile_m = gemm_p.getSetting("sg_tile_m");
        auto sg_tile_n = gemm_p.getSetting("sg_tile_n");

        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(3);

        auto input_layout = params.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT);
        auto experts_weight_layout = params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
        auto input_tokens_lens_layout = params.get_input_layout(4);
        auto output_layout = params.get_output_layout();

        // input : [num_actual_experts * n, k]
        size_t n = input_layout.get_shape()[0];
        // experts_weight : [num_experts, m, k]
        size_t m = experts_weight_layout.get_shape()[1];
        size_t k = experts_weight_layout.get_shape()[2];
        std::cout << "m : " << m << " n : " << n << " k : " << k << std::endl;
        wgs.local = { sg_per_wg_m * get_subgroup_size(device_info.arch),
                      sg_per_wg_n,
                      1};
        wgs.global = { align_to(ceil_div(m, sg_tile_m), sg_per_wg_m) * get_subgroup_size(device_info.arch),
                       align_to(ceil_div(n, sg_tile_n), sg_per_wg_n),
                       static_cast<size_t>(rtp->num_actual_used_experts)};
        std::cout << "output layout : " << output_layout.to_short_string() << std::endl;
        std::cout << "gws : " << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << std::endl;
        std::cout << "lws : " << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << std::endl;
        ScalarDescriptor s_m{ScalarDescriptor::Types::INT32};
        s_m.v.s32 = m;
        scalars.push_back(s_m);

        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = k;
        scalars.push_back(s_k);
    }};
}

std::string MoEGemmMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
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
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});   // experts_ids
    args.push_back({ArgumentDescriptor::Types::INPUT, 3});   // input_offset_per_expert

    args.push_back({ArgumentDescriptor::Types::INPUT, 4});   // n_array

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    args.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});

    if (params.input_layouts[1].data_type == data_types::u4 ||
        params.input_layouts[1].data_type == data_types::i4) {
        args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // weight scales // TODO adjust idx
        args.push_back({ArgumentDescriptor::Types::INPUT, 6});  // weight zp
    }

    return args;
}

KernelData MoEGemmMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    std::cout << "start get kernel data for micro " << get_kernel_name() << std::endl;
    micro::Package moe_gemm;
    const auto& device_info = params.get_device_info();
    init_microkernels(params, moe_gemm, m_is_prefill); // TODO

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
    shim_options.subgroupSize = static_cast<int32_t>(get_subgroup_size(device_info.arch));
    shim_options.useTileOps = true;
    shim_options.decorator = "moe";

    kd.code->jit += generateShim(moe_gemm, micro::HostLanguage::OpenCL_C, shim_options);
    if (moe_gemm.grfMin > 128) {
       kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(moe_gemm));

    // Micro kernel is using slm implicitly inside the kernel.
    // Therefore the slm should be allocated.
    auto slm_size = kd.micro_kernels[0]->p.getSetting("slm_size");
    kd.params.local_memory_args.clear();
    kd.params.local_memory_args.push_back(slm_size > 0 ? slm_size : 1);
    std::cout << "get_kernel_data finished" << std::endl;
    return kd;
}
}  // namespace ov::intel_gpu::ocl
#endif
