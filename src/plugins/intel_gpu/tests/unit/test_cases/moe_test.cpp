// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_mask_gen.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>

using namespace cldnn;
using namespace ::tests;


TEST(moe_unit, moe_mask_gen_test) {
    auto& engine = get_test_engine();

    // num experts 32
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
    for (size_t i = 0; i < output->get_layout().count(); i++)
        std::cout << "[" << i << "] " << output_ptr[i] << std::endl;
}
