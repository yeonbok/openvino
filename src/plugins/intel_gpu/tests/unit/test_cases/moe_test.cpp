// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_mask_gen.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include <intel_gpu/primitives/moe_gemm.hpp>

using namespace cldnn;
using namespace ::tests;


TEST(moe_unit, moe_mask_gen_test) {
    auto& engine = get_test_engine();

    // num total experts 32
    // num active experts 2
    // input activation [30, 64]
    // topk [30, 2]
    // output expert_info_offsets [32]
    // output tokens_indices_per_expert [30*2]

    std::vector<int32_t> topk_idx = {
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
    };

    std::vector<int32_t> dummy_data = {
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8
    };

//    std::vector<int32_t> gather_info_data = {
//            // expert offsets
//            -1, -1, -1, -1, 0,  -1, -1, -1,
//            30, -1, -1, -1, -1, -1, -1, -1,
//            -1, -1, -1, -1, -1, -1, -1, -1,
//            -1, -1, -1, -1, -1, -1, -1, -1,
//            // tokens per experts 
//            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
//            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
//            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
//            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
//    };

    std::vector<int32_t> gather_info_data =
    {
        -1, -1, -1, -1, 0, -1, -1, -1,
        30, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        0,   0,  0,  0, 30,  0,  0,  0, 30,  0,  0,  0,
        0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,   0,  0,  0,  0,  0,  0,  0,
        0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
       16,  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
       16,  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    };

    auto topk_shape = ov::PartialShape{ov::Dimension(30), ov::Dimension(2)};
    auto topk_idx_layout = layout{topk_shape, data_types::i32, format::bfyx};
    auto topk_idx_mem = engine.allocate_memory(topk_idx_layout);

    // dummy now
    auto weight_data = engine.allocate_memory(topk_idx_layout);
    auto weight_layout = layout{topk_shape, data_types::i32, format::bfyx};
    auto weight_mem = engine.allocate_memory(topk_idx_layout);

    set_values(topk_idx_mem, topk_idx);
    topology topology(
        input_layout("input_topk", topk_idx_layout),
        data("weight_dummy", weight_mem),
        moe_mask_gen("moe_mask_gen", input_info("input_topk"), input_info("weight_dummy"), 32, 2)
    );
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_topk", topk_idx_mem);
    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();

    cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output->get_layout().count(); i++) {
//        std::cout << output_ptr[i] << ", ";
        ASSERT_EQ(output_ptr[i], gather_info_data[i]);
    }
    std::cout << std::endl;
}

TEST(moe_unit, moe_gather_test) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    // num total experts 32
    // num active experts 2
    // input activation [30, 64]
    // mask_gather_info
    //    expert_info_offsets [32]
    //    tokens_indices_per_expert [30*2]
    size_t num_tokens = 30;
    size_t hidden_size = 64;
    size_t num_total_experts = 32;
    size_t num_active_experts = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto gather_info_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto gather_info_layout = layout{gather_info_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        input_layout("gather_info", gather_info_layout),
        moe_gather("moe_gather", input_info("input"), input_info("gather_info"), num_total_experts, num_active_experts)
    );

    auto input_data = rg.generate_random_1d<ov::float16>(num_tokens * hidden_size, -1, 1);
    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    std::vector<int32_t> gather_info_data =
    {
        // experts offset
        -1, -1, -1, -1, 0, -1, -1, -1,
        30, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        // experts number
        0,   0,  0,  0, 30,  0,  0,  0, 30,  0,  0,  0,
        0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,   0,  0,  0,  0,  0,  0,  0,
        // tokens per expert
        0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
       16,  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
       16,  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    };

    size_t gather_info_data_size = num_total_experts + num_tokens * num_active_experts;
    auto gather_info_data_shape = ov::PartialShape{ov::Dimension(gather_info_data_size)};
    auto gather_info_data_layout = layout{gather_info_data_shape, data_types::f16, format::bfyx};
    auto gather_info_mem = engine.allocate_memory(gather_info_data_layout);
    set_values(gather_info_mem, gather_info_data);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);
    network.set_input_data("gather_info", gather_info_mem);
    auto outputs = network.execute();

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < num_tokens * hidden_size; i++) {
        ASSERT_EQ(input_data[i], output_ptr[i]);
    }
}

TEST(moe_unit, moe_gemm_test) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    // num total experts 32
    // num active experts 2
    // input activation [30, 64]
    // mask_gather_info
    //    expert_info_offsets [32]
    //    tokens_indices_per_expert [30*2]
    size_t num_tokens = 10;
    size_t hidden_size = 16;
    size_t num_total_experts = 4;
    size_t experts_out_N = 16;
    size_t num_actual_experts = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    //    auto experts_data = rg.generate_random_1d<ov::float16>(num_total_experts * hidden_size * experts_out_N, -1, 1);
    // weight to fill with 1.0f for initial test
    std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
    set_values(experts_mem, experts_data);

    auto input_offset_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offsets_layout = layout{input_offset_shape, data_types::i32, format::bfyx};
    auto weights_offset_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto weight_offsets_layout = layout{weights_offset_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};
    auto output_offset_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto output_offsets_layout = layout{output_offset_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        data("moe_experts", experts_mem),
        input_layout("input_offsets", input_offsets_layout),
        input_layout("weight_offsets", weight_offsets_layout),
        input_layout("output_offsets", output_offsets_layout),
        input_layout("input_tokens_lens", input_tokens_lens_layout),
        moe_gemm("moe_gemm", input_info("input"),
                             input_info("moe_experts"),
                             input_info("input_offsets"),
                             input_info("weight_offsets"),
                             input_info("output_offsets"),
                             input_info("input_tokens_lens"))
    );

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data(num_total_experts * num_tokens * hidden_size, 0.1f);

    set_values(input_mem, input_data);
    std::vector<int32_t> input_offset_data = {0, 16*10*2};
    std::vector<int32_t> weight_offset_data = {0, 16*16*2};
    std::vector<int32_t> output_offset_data = {0, 16*10*2};
    std::vector<int32_t> input_tokens_lens = {3, 7};

    auto input_offset_data_shape = ov::PartialShape{ov::Dimension(num_actual_experts)};
    auto weights_offset_data_shape = ov::PartialShape{ov::Dimension(num_actual_experts)};
    auto output_offset_data_shape = ov::PartialShape{ov::Dimension(num_actual_experts)};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(num_actual_experts)};

    auto input_offsets_data_layout = layout{input_offset_data_shape, data_types::i32, format::bfyx};
    auto weight_offsets_data_layout = layout{weights_offset_data_shape, data_types::i32, format::bfyx};
    auto output_offsets_data_layout = layout{output_offset_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};

    auto input_offset_mem = engine.allocate_memory(input_offsets_data_layout);
    auto weight_offset_mem = engine.allocate_memory(weight_offsets_data_layout);
    auto output_offset_mem = engine.allocate_memory(output_offsets_data_layout);
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);

    set_values(input_offset_mem, input_offset_data);
    set_values(weight_offset_mem, weight_offset_data);
    set_values(output_offset_mem, output_offset_data);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("input_offsets", input_offset_mem);
    network.set_input_data("weight_offsets", weight_offset_mem);
    network.set_input_data("output_offsets", output_offset_mem);
    network.set_input_data("input_tokens_lens", input_tokens_lens_mem);

    auto outputs = network.execute();

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::cout << output_ptr[0] << std::endl;
}
