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
#include "intel_gpu/primitives/fully_connected.hpp"

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
//    network network(engine, topology, get_test_default_config(engine));
    auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
    network->set_input_data("input_topk", topk_idx_mem);
    auto outputs = network->execute();
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
    // num_tokens 30
    // hidden_size 64
    // num total experts 32
    // experts_per_token 2
    // num_actual_used_experts 7
    // input0 activation [30, 64]
    // input1 experts_info_offset [7] 
    // input2 tokens_per_expert [30*2*64] 
    size_t num_tokens = 30;
    size_t num_total_experts = 32; 
    size_t hidden_size = 64;
    int32_t num_experts_per_token = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto experts_info_offsets_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_info_offsets_layout = layout{experts_info_offsets_shape, data_types::i32, format::bfyx};

    auto tokens_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto tokens_per_expert_layout = layout{tokens_per_expert_shape, data_types::i32, format::bfyx};

    auto tokens_len_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto tokens_len_per_expert_layout = layout{tokens_len_per_expert_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        input_layout("experts_info_offsets", experts_info_offsets_layout),
        input_layout("tokens_per_expert", tokens_per_expert_layout),
        input_layout("tokens_len_per_expert", tokens_len_per_expert_layout),
        moe_gather("moe_gather", input_info("input"), input_info("experts_info_offsets"), input_info("tokens_per_expert"), input_info("tokens_len_per_expert"), num_experts_per_token)
    );

    std::vector<ov::float16> input_data;
    for (size_t i = 0; i < num_tokens; ++i) {
        for (size_t h = 0; h < hidden_size; ++h)
            input_data.push_back(static_cast<ov::float16>(i));
    }
    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    // topk result
    std::vector<std::vector<int32_t>> experts_per_token = {{0, 5},  {5, 7},   {0, 10}, {11, 20}, {7, 10},  {0, 7},   {20, 31}, {11, 31}, {11, 20}, {7, 10},
                                          {0, 5},  {11, 31}, {0, 7},  {0, 20},  {10, 31}, {10, 20}, {7, 31},  {0, 31},  {5, 31},  {7, 31},
                                          {7, 20}, {0, 10},  {0, 5},  {5, 11},  {7, 11},  {5, 31},  {7, 31},  {0, 31},  {0, 10},  {11, 20}};
    
    std::vector<std::vector<int32_t>> tokens_per_expert_tmp(num_total_experts, std::vector<int32_t>{});

    for (size_t i = 0; i < experts_per_token.size(); ++i) {
        for (size_t j = 0; j < experts_per_token[i].size(); ++j)
            tokens_per_expert_tmp[experts_per_token[i][j]].push_back(i);
    }

    std::vector<int32_t> experts_info_offsets_data;
    std::vector<int32_t> tokens_per_expert_data;
    std::vector<int32_t> tokens_len_per_expert_data;

    for (size_t i = 0; i < tokens_per_expert_tmp.size(); ++i) {
        if (tokens_per_expert_tmp[i].empty())
            continue;
        experts_info_offsets_data.push_back(static_cast<int32_t>(tokens_per_expert_data.size()));
        for (size_t j = 0; j < tokens_per_expert_tmp[i].size(); ++j) {
            tokens_per_expert_data.push_back(tokens_per_expert_tmp[i][j]);
        }
        tokens_len_per_expert_data.push_back(static_cast<int32_t>(tokens_per_expert_tmp[i].size()));
    }
    // experts 0, 5, 7, 10, 11, 20, 31 are used
    // experts[0] offset  : 0  {0, 2, 5, 10, 12, 13, 17, 21, 22, 27, 28}
    // experts[5] offset  : 11 {0, 1, 10, 18, 22, 23, 25}
    // experts[7] offset  : 18 {1, 4, 5, 9, 12, 16, 19, 20, 24, 26}
    // experts[10] offset : 28 {2, 4, 9, 14, 15, 21, 28}
    // experts[11] offset : 35 {3, 7, 8, 11, 23, 24, 29}
    // experts[20] offset : 42 {3, 6, 8, 13, 15, 20, 29}
    // experts[31] offset : 49 {6, 7, 11, 14, 16, 17, 18, 19, 25, 26, 27}


    auto experts_info_offsets_data_shape = ov::PartialShape{ov::Dimension(experts_info_offsets_data.size())};
    auto experts_info_offsets_data_layout = layout{experts_info_offsets_data_shape, data_types::i32, format::bfyx};
    auto experts_info_offsets_mem = engine.allocate_memory(experts_info_offsets_data_layout);
    set_values(experts_info_offsets_mem, experts_info_offsets_data);

    auto tokens_per_expert_data_shape    = ov::PartialShape{ov::Dimension(tokens_per_expert_data.size())};
    auto tokens_per_expert_data_layout   = layout{tokens_per_expert_data_shape, data_types::i32, format::bfyx};
    auto tokens_per_expert_data_mem      = engine.allocate_memory(tokens_per_expert_data_layout);
    set_values(tokens_per_expert_data_mem, tokens_per_expert_data);

    auto tokens_len_per_expert_data_shape    = ov::PartialShape{ov::Dimension(tokens_len_per_expert_data.size())};
    auto tokens_len_per_expert_data_layout   = layout{tokens_len_per_expert_data_shape, data_types::i32, format::bfyx};
    auto tokens_len_per_expert_data_mem      = engine.allocate_memory(tokens_len_per_expert_data_layout);
    set_values(tokens_len_per_expert_data_mem, tokens_len_per_expert_data);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_mem);
    network.set_input_data("experts_info_offsets", experts_info_offsets_mem);
    network.set_input_data("tokens_per_expert", tokens_per_expert_data_mem);
    network.set_input_data("tokens_len_per_expert", tokens_len_per_expert_data_mem);
    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<ov::float16> ref_output;
    for (size_t i = 0; i < tokens_per_expert_data.size(); ++i) {
        int32_t token_id = tokens_per_expert_data[i];
        for (size_t h = 0; h < hidden_size; ++h) {
            ref_output.push_back(input_data[token_id * hidden_size + h]);
        }
    }
    for (size_t i = 0; i < num_tokens * num_experts_per_token * hidden_size; i++) {
        ASSERT_EQ(ref_output[i], output_ptr[i]);
    }
}

static std::vector<ov::float16> get_ref_moe_gemm(std::vector<ov::float16>& input, std::vector<ov::float16>& experts, 
                                    size_t M, size_t K, size_t N,
                                    std::vector<int32_t>& experts_ids, std::vector<int32_t>& input_offset_per_expert,
                                    std::vector<int32_t>& input_tokens_lens,
                                    size_t num_active_experts_per_token,
                                    bool is_prefill = true) {
    std::cout << "Generte ref code for " << (is_prefill ? "prefill " : "generate ") << " phase" << std::endl;
    std::vector<ov::float16> output(M * num_active_experts_per_token * N, 0.0f);
    size_t input_stride = K;
    size_t expert_stride = K * N;
    size_t output_stride = N;
    for (size_t i = 0; i < input_offset_per_expert.size(); i++) {
        int32_t expert_id = experts_ids[i];
        int32_t input_offset = 0;
        if (is_prefill)
            input_offset = input_offset_per_expert[i] * input_stride;
        int32_t weight_offset = expert_id * expert_stride;
        int32_t output_offset = input_offset_per_expert[i] * output_stride;
        int32_t tokens_lens = input_tokens_lens[i];
        for (int32_t m = 0; m < tokens_lens; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += input[input_offset + m * input_stride + k] * experts[weight_offset + n * K + k];
                }
                output[output_offset + m * output_stride + n] = sum;
            }
        }
    }
    return output;
};

TEST(moe_unit, moe_gemm_test_small) {
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
    int32_t num_active_experts_per_token = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    // weight to fill with 1.0f for initial test
    std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
    set_values(experts_mem, experts_data);

    auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        data("moe_experts", experts_mem),
        input_layout("experts_ids", experts_ids_layout),
        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
        input_layout("input_tokens_lens", input_tokens_lens_layout),
        moe_gemm("moe_gemm", input_info("input"),
                             input_info("moe_experts"),
                             input_info("experts_ids"),
                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
                             input_info("input_tokens_lens"),
                             num_active_experts_per_token
        )
    );

    std::vector<int32_t> input_tokens_lens (num_total_experts, -1);
    input_tokens_lens[0] = 3;
    input_tokens_lens[1] = 7;

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data(num_tokens * hidden_size);
    for (size_t i = 0; i < input_tokens_lens[0] * hidden_size; ++i) {
        input_data[i] = 1.0f;
    }
    for (size_t i = input_tokens_lens[0] * hidden_size; i <  num_tokens * hidden_size; ++i) {
        input_data[i] = 2.0f;
    }
    set_values(input_mem, input_data);

    std::vector<int32_t> experts_ids_data(num_total_experts, -1);
    experts_ids_data[0] = 0;
    experts_ids_data[1] = 2;
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 3};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);  

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("experts_ids", experts_ids_mem);
    network.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network.set_input_data("input_tokens_lens", input_tokens_lens_mem);

    auto outputs = network.execute();
    auto output_ref = get_ref_moe_gemm(input_data, experts_data, 
                                    num_tokens, hidden_size, experts_out_N,
                                    experts_ids_data, input_offset_per_expert_data,
                                    input_tokens_lens, num_active_experts_per_token);

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t m = 0; m < num_tokens; m++) {
        for (size_t n = 0; n < experts_out_N; n++) {
            //std::cout << "c[" << m << "][" << n << "]: " << (float)output_ptr[m * experts_out_N + n] << std::endl;
            ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
        }
    }
}

TEST(moe_unit, moe_gemm_test_large) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        auto& engine = get_test_engine();
        size_t num_tokens = 100;
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = 16;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(num_active_experts_per_token * num_tokens * hidden_size);
        int cur_token_base = 0;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            for (int len = 0; len < input_tokens_lens[i]; ++len) {
                for (size_t h = 0; h < hidden_size; ++h) {
                    input_data[(cur_token_base + len) * hidden_size + h] = static_cast<ov::float16>((i + 1) / 10.0f);
                }
            }
            cur_token_base += input_tokens_lens[i];
        }

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                //            std::cout << "c[" << m << "][" << n << "]: " << (float)output_ptr[m * experts_out_N + n] << std::endl;
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 100;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
        //network.set_input_data("input", input_prim);
    }
}

TEST(moe_unit, moe_gemm_test_generate_up) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        auto& engine = get_test_engine();
        size_t num_tokens = 1;
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = num_active_experts_per_token;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
            std::cout << "input_tokens_lens[" << i << "] : " << input_tokens_lens[i] << std::endl;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(1 * hidden_size);

        for (size_t h = 0; h < hidden_size; ++h) {
            input_data[h] = static_cast<ov::float16>((num_tokens) / 10.0f);
        }
        std::cout << "input shape : " << input_data_shape.to_string() << std::endl;

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
            std::cout << "input_offset_per_data[" << i << "] : " << input_offset_per_expert_data[i] << std::endl;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        std::cout << "input_offset_per_expert_data_shape : " << input_offset_per_expert_data_shape.to_string() << std::endl;
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token, 
                                           false);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 1;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
    }
}

TEST(moe_unit, moe_gemm_test_generate_down) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        size_t num_tokens = 1;
        auto& engine = get_test_engine();
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = num_active_experts_per_token;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
            std::cout << "input_tokens_lens[" << i << "] : " << input_tokens_lens[i] << std::endl;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(num_active_experts_per_token * hidden_size);

        for (int e = 0; e < num_active_experts_per_token; ++e) {
            for (size_t h = 0; h < hidden_size; ++h) {
                input_data[e * hidden_size + h] = static_cast<ov::float16>((1 + num_tokens * e) / 10.0f);
            }
        }
        std::cout << "input shape : " << input_data_shape.to_string() << std::endl;

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
            std::cout << "experts_ids_data [" << i << "] : " << experts_ids_data[i] << std::endl;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
            std::cout << "input_offset_per_data[" << i << "] : " << input_offset_per_expert_data[i] << std::endl;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        std::cout << "input_offset_per_expert_data_shape : " << input_offset_per_expert_data_shape.to_string() << std::endl;
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token, 
                                           true);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens * num_active_experts_per_token; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 1;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
    }
}
