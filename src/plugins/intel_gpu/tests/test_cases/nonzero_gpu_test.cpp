// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/non_zero.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>
#include <gtest/gtest.h>

using namespace cldnn;
using namespace ::tests;

inline void DoCountNonzeroTest(engine& engine,
                               const cldnn::memory::ptr& input_data,
                               const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(
        count_nonzero("count_nonzero", "InputData")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    auto outputs = network.execute();
    auto output = outputs.at("count_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

inline void DoGatherNonzeroTest(engine& engine,
                                const cldnn::memory::ptr& input_data,
                                const cldnn::memory::ptr& output_shape,
                                const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(input_layout("OutputShape", output_shape->get_layout()));
    topology.add(
        gather_nonzero("gather_nonzero", "InputData", "OutputShape")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    network.set_input_data("OutputShape", output_shape);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    cldnn::mem_lock<int32_t> shape_ptr(output_shape, get_test_stream());

    int num_ranks = shape_ptr[0];
    int num_nonzero = shape_ptr[1];

    for (int i = 0; i < num_nonzero; i++) {
        bool found = false;
        for (int j = 0; j < num_nonzero; j++) {
            for (int k = 0; k < num_ranks; k++) {
                if (output_ptr[i+num_nonzero*k] != expected_results[j+num_nonzero*k])
                    break;

                if (k == (num_ranks - 1)) {
                    found = true;
                }
            }
            if (found)
                break;
        }

        EXPECT_TRUE(found);
    }

    // for (size_t i = 0; i < expected_results.size(); ++i) {
    //     EXPECT_EQ(expected_results[i], output_ptr[i]);
    // }
}

inline void DoNonzeroTest(engine& engine,
                          const cldnn::memory::ptr& input_data,
                          const std::vector<int32_t>& expected_shape,
                          const std::vector<int32_t>& expected_results)
{
    topology topology;
    topology.add(input_layout("InputData", input_data->get_layout()));
    topology.add(
        count_nonzero("count_nonzero", "InputData")
    );
    topology.add(
        gather_nonzero("gather_nonzero", "InputData", "count_nonzero")
    );

    network network(engine, topology);

    network.set_input_data("InputData", input_data);
    auto outputs = network.execute();
    auto output = outputs.at("gather_nonzero").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    int num_ranks = expected_shape[0];
    int num_nonzero = expected_shape[1];

    for (int i = 0; i < num_nonzero; i++) {
        bool found = false;
        for (int j = 0; j < num_nonzero; j++) {
            for (int k = 0; k < num_ranks; k++) {
                if (output_ptr[i+num_nonzero*k] != expected_results[j+num_nonzero*k])
                    break;
                    
                if (k == (num_ranks - 1)) {
                    found = true;
                }
            }
            if (found)
                break;
        }

        EXPECT_TRUE(found);
    }

    // for (size_t i = 0; i < expected_results.size(); ++i) {
    //     EXPECT_EQ(expected_results[i], output_ptr[i]);
    // }
}

TEST(count_nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(2),
        FLOAT16(7), FLOAT16(10), FLOAT16(4),
    });

    std::vector<int32_t> expected_results = {
        4,  8,  1,  1,
    };

    DoCountNonzeroTest(engine, input, expected_results);
}

TEST(gather_nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });
    auto output_shape = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 4, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(2),
        FLOAT16(7), FLOAT16(10), FLOAT16(4),
    });

    set_values(output_shape, {
        4,  8,  1,  1,
    });

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  1,  2,  2,  2,
        0,  0,  0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  2,  0,  1,  2,
    };

    DoGatherNonzeroTest(engine, input, output_shape, expected_results);
}

TEST(gather_nonzero_gpu_fp16, test2) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });
    auto output_shape = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 1, 4, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(0),
        FLOAT16(7), FLOAT16(0), FLOAT16(4),
    });

    set_values(output_shape, {
        4,  6,  1,  1,
    });

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  2,  2,
        0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  0,  2,
    };

    DoGatherNonzeroTest(engine, input, output_shape, expected_results);
}

TEST(nonzero_gpu_fp16, test1) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 3, 1 } });

    set_values(input, {
        FLOAT16(0), FLOAT16(1), FLOAT16(8),
        FLOAT16(5), FLOAT16(5), FLOAT16(0),
        FLOAT16(7), FLOAT16(0), FLOAT16(4),
    });

    std::vector<int32_t> expected_shape = {
        4,  6,  1, 1,
    };

    std::vector<int32_t> expected_results = {
        0,  0,  0,  0,  0,  0,
        0,  0,  1,  1,  2,  2,
        0,  0,  0,  0,  0,  0,
        1,  2,  0,  1,  0,  2,
    };

    DoNonzeroTest(engine, input, expected_shape, expected_results);
}

//TEST(gather_elements_gpu_fp16, d3283_i2283_a0) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_b;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 3, 2, 8, 3 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 8, 3 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(5), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(4), FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(7), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(8), FLOAT16(3), FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(5), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(0), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(3),
//        FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(10), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//        FLOAT16(9), FLOAT16(1), FLOAT16(0), FLOAT16(7), FLOAT16(9), FLOAT16(6), FLOAT16(8), FLOAT16(7),
//        FLOAT16(10), FLOAT16(9), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(5), FLOAT16(6), FLOAT16(9),
//        FLOAT16(4), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(1),
//        FLOAT16(1), FLOAT16(6), FLOAT16(8), FLOAT16(0), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(8),
//        FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(9), FLOAT16(1), FLOAT16(2), FLOAT16(7), FLOAT16(1),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(7), FLOAT16(10), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(1), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(10), FLOAT16(6), FLOAT16(1),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(4), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(1), FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(0), FLOAT16(5), FLOAT16(3),
//        FLOAT16(6), FLOAT16(5), FLOAT16(6), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(6), FLOAT16(1),
//        FLOAT16(3), FLOAT16(5), FLOAT16(5), FLOAT16(4), FLOAT16(4), FLOAT16(7), FLOAT16(8), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(7), FLOAT16(9), FLOAT16(8), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(1), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(0), FLOAT16(6), FLOAT16(1),
//        FLOAT16(2), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(2), FLOAT16(3), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(0), FLOAT16(5), FLOAT16(0), FLOAT16(5), FLOAT16(3),
//        FLOAT16(6), FLOAT16(9), FLOAT16(2), FLOAT16(0), FLOAT16(4), FLOAT16(2), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(10), FLOAT16(7), FLOAT16(10), FLOAT16(2),
//        FLOAT16(9), FLOAT16(3), FLOAT16(0), FLOAT16(7), FLOAT16(6), FLOAT16(8), FLOAT16(8), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(3), FLOAT16(3), FLOAT16(10), FLOAT16(6), FLOAT16(1),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 8, 3), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d2235_i2235_a3) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_x;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 3, 5 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 3, 5 } }); // indices
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8),
//        FLOAT16(5), FLOAT16(5), FLOAT16(2),
//        FLOAT16(0), FLOAT16(7), FLOAT16(7),
//        FLOAT16(10), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(0),
//        FLOAT16(5), FLOAT16(7), FLOAT16(0),
//        FLOAT16(4), FLOAT16(0), FLOAT16(4),
//        FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1),
//        FLOAT16(7), FLOAT16(4), FLOAT16(7),
//        FLOAT16(10), FLOAT16(8), FLOAT16(2),
//        FLOAT16(0), FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10),
//        FLOAT16(4), FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8), FLOAT16(7),
//        FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8),
//        FLOAT16(5), FLOAT16(2), FLOAT16(3),
//        FLOAT16(3), FLOAT16(1), FLOAT16(5),
//        FLOAT16(9), FLOAT16(10), FLOAT16(0),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8),
//        FLOAT16(2), FLOAT16(2), FLOAT16(5),
//        FLOAT16(0), FLOAT16(0), FLOAT16(7),
//        FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(0), FLOAT16(9), FLOAT16(0),
//        FLOAT16(7), FLOAT16(0), FLOAT16(7),
//        FLOAT16(4), FLOAT16(0), FLOAT16(4),
//        FLOAT16(6), FLOAT16(7), FLOAT16(10),
//        FLOAT16(5), FLOAT16(9), FLOAT16(5),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(8), FLOAT16(2), FLOAT16(2),
//        FLOAT16(8), FLOAT16(8), FLOAT16(8),
//        FLOAT16(8), FLOAT16(6), FLOAT16(10),
//        FLOAT16(4), FLOAT16(10), FLOAT16(10),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(0), FLOAT16(0), FLOAT16(9),
//        FLOAT16(4), FLOAT16(8), FLOAT16(8),
//        FLOAT16(3), FLOAT16(3), FLOAT16(5),
//        FLOAT16(5), FLOAT16(3), FLOAT16(3),
//        FLOAT16(9), FLOAT16(9), FLOAT16(0),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 3, 5), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d1329_i1359_an1) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_x;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 2, 9 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 3, 5, 9 } }); // indices
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1),
//        FLOAT16(8), FLOAT16(5),
//        FLOAT16(5), FLOAT16(2),
//        FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10),
//        FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0),
//        FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0),
//        FLOAT16(4), FLOAT16(0),
//        FLOAT16(4), FLOAT16(7),
//        FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5),
//        FLOAT16(1), FLOAT16(7),
//        FLOAT16(4), FLOAT16(7),
//        FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0),
//        FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8),
//        FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8),
//        FLOAT16(7), FLOAT16(0),
//        FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4),
//        FLOAT16(8), FLOAT16(5),
//        FLOAT16(2), FLOAT16(3),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(8),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(5),
//        FLOAT16(0), FLOAT16(7), FLOAT16(0), FLOAT16(7), FLOAT16(7),
//        FLOAT16(10), FLOAT16(7), FLOAT16(7), FLOAT16(10), FLOAT16(10),
//        FLOAT16(4), FLOAT16(4), FLOAT16(5), FLOAT16(4), FLOAT16(4),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(9), FLOAT16(9),
//        FLOAT16(5), FLOAT16(0), FLOAT16(0), FLOAT16(5), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(4), FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(4), FLOAT16(7),
//        FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(10),
//        FLOAT16(5), FLOAT16(9), FLOAT16(5), FLOAT16(9), FLOAT16(5),
//        FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(1), FLOAT16(7),
//        FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(7), FLOAT16(7),
//        FLOAT16(8), FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(8),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(8), FLOAT16(8), FLOAT16(3), FLOAT16(8), FLOAT16(8),
//        FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(8), FLOAT16(6),
//        FLOAT16(10), FLOAT16(4), FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(10), FLOAT16(2), FLOAT16(2), FLOAT16(10), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(0), FLOAT16(7), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(6), FLOAT16(9), FLOAT16(9), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(2), FLOAT16(4), FLOAT16(2), FLOAT16(4),
//        FLOAT16(5), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(8),
//        FLOAT16(3), FLOAT16(3), FLOAT16(2), FLOAT16(3), FLOAT16(3),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(1, 3, 5, 9), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d12853_i12923_a3) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_y;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 1, 2, 8, 5, 3 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 1, 2, 8, 2, 3 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(5), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(4), FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(7), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(8), FLOAT16(3), FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(5), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(0), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(3),
//        FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(10), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//        FLOAT16(9), FLOAT16(1), FLOAT16(0), FLOAT16(7), FLOAT16(9), FLOAT16(6), FLOAT16(8), FLOAT16(7),
//        FLOAT16(10), FLOAT16(9), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(5), FLOAT16(6), FLOAT16(9),
//        FLOAT16(4), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(1),
//        FLOAT16(1), FLOAT16(6), FLOAT16(8), FLOAT16(0), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(8),
//        FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(9), FLOAT16(1), FLOAT16(2), FLOAT16(7), FLOAT16(1),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(7), FLOAT16(10), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(1), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(10), FLOAT16(6), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1), FLOAT16(6), FLOAT16(2), FLOAT16(5), FLOAT16(5), FLOAT16(10),
//        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(6), FLOAT16(1), FLOAT16(7), FLOAT16(6), FLOAT16(8),
//        FLOAT16(2), FLOAT16(5), FLOAT16(4), FLOAT16(2), FLOAT16(0), FLOAT16(9), FLOAT16(4), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1), FLOAT16(9), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(4),
//        FLOAT16(2), FLOAT16(1), FLOAT16(8), FLOAT16(5), FLOAT16(3), FLOAT16(4), FLOAT16(8), FLOAT16(10),
//        FLOAT16(7), FLOAT16(2), FLOAT16(7), FLOAT16(9), FLOAT16(2), FLOAT16(9), FLOAT16(5), FLOAT16(5),
//        FLOAT16(6), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(10), FLOAT16(6), FLOAT16(4), FLOAT16(9),
//        FLOAT16(7), FLOAT16(7), FLOAT16(10), FLOAT16(10), FLOAT16(9), FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(1), FLOAT16(4), FLOAT16(6), FLOAT16(9), FLOAT16(4), FLOAT16(8), FLOAT16(9), FLOAT16(7),
//        FLOAT16(8), FLOAT16(7), FLOAT16(8), FLOAT16(0), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(0),
//        FLOAT16(7), FLOAT16(5), FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(10), FLOAT16(9), FLOAT16(9),
//        FLOAT16(5), FLOAT16(1), FLOAT16(4), FLOAT16(10), FLOAT16(2), FLOAT16(4), FLOAT16(3), FLOAT16(5),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(2), FLOAT16(4), FLOAT16(3), FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(4), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(3), FLOAT16(1), FLOAT16(4), FLOAT16(2), FLOAT16(4), FLOAT16(2), FLOAT16(1), FLOAT16(3),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(4), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(3),
//        FLOAT16(4), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(3), FLOAT16(4), FLOAT16(3), FLOAT16(4), FLOAT16(4), FLOAT16(1), FLOAT16(0), FLOAT16(3),
//        FLOAT16(2), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(4), FLOAT16(3), FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(3), FLOAT16(4), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(3), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(3), FLOAT16(3), FLOAT16(4), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(3),
//        FLOAT16(3), FLOAT16(4), FLOAT16(3), FLOAT16(3), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(3),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(4), FLOAT16(0), FLOAT16(4),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(0), FLOAT16(8), FLOAT16(7), FLOAT16(6), FLOAT16(2), FLOAT16(0), FLOAT16(5),
//        FLOAT16(2), FLOAT16(1), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(2), FLOAT16(0), FLOAT16(5),
//        FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(0), FLOAT16(10), FLOAT16(5), FLOAT16(3), FLOAT16(4),
//        FLOAT16(5), FLOAT16(4), FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(6), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(7),
//        FLOAT16(1), FLOAT16(9), FLOAT16(8), FLOAT16(9), FLOAT16(1), FLOAT16(5), FLOAT16(8), FLOAT16(8),
//        FLOAT16(5), FLOAT16(2), FLOAT16(3), FLOAT16(6), FLOAT16(1), FLOAT16(7), FLOAT16(6), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0), FLOAT16(6), FLOAT16(2), FLOAT16(7), FLOAT16(6), FLOAT16(1),
//        FLOAT16(7), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(0), FLOAT16(9), FLOAT16(0), FLOAT16(4),
//        FLOAT16(2), FLOAT16(2), FLOAT16(7), FLOAT16(5), FLOAT16(3), FLOAT16(9), FLOAT16(4), FLOAT16(5),
//        FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(7), FLOAT16(4), FLOAT16(8), FLOAT16(5), FLOAT16(9),
//        FLOAT16(1), FLOAT16(7), FLOAT16(10), FLOAT16(0), FLOAT16(9), FLOAT16(4), FLOAT16(5), FLOAT16(5),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 8, 2, 3), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d25441_i22441_an4) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_f;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 2, 5, 4, 4, 1 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 2, 2, 4, 4, 1 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(5),
//        FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(4), FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1), FLOAT16(7),
//        FLOAT16(4), FLOAT16(7), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(8),
//        FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(5),
//        FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(0),
//        FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(3),
//        FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0),
//        FLOAT16(10), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5), FLOAT16(5),
//        FLOAT16(10), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//        FLOAT16(9), FLOAT16(1), FLOAT16(0), FLOAT16(7),
//        FLOAT16(9), FLOAT16(6), FLOAT16(8), FLOAT16(7),
//        FLOAT16(10), FLOAT16(9), FLOAT16(2), FLOAT16(3),
//        FLOAT16(3), FLOAT16(5), FLOAT16(6), FLOAT16(9),
//        FLOAT16(4), FLOAT16(9), FLOAT16(2), FLOAT16(4),
//        FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(1),
//        FLOAT16(1), FLOAT16(6), FLOAT16(8), FLOAT16(0),
//        FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(8),
//        FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(9),
//        FLOAT16(1), FLOAT16(2), FLOAT16(7), FLOAT16(1),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0), FLOAT16(4),
//        FLOAT16(0), FLOAT16(7), FLOAT16(10), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7),
//        FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(1), FLOAT16(6), FLOAT16(9),
//        FLOAT16(6), FLOAT16(10), FLOAT16(6), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1), FLOAT16(6),
//        FLOAT16(2), FLOAT16(5), FLOAT16(5), FLOAT16(10),
//        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(6),
//        FLOAT16(1), FLOAT16(7), FLOAT16(6), FLOAT16(8),
//
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(2), FLOAT16(4), FLOAT16(3),
//        FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(4), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(3), FLOAT16(1), FLOAT16(4), FLOAT16(2),
//        FLOAT16(4), FLOAT16(2), FLOAT16(1), FLOAT16(3),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(4),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(3),
//        FLOAT16(4), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(3), FLOAT16(4), FLOAT16(3), FLOAT16(4),
//        FLOAT16(4), FLOAT16(1), FLOAT16(0), FLOAT16(3),
//        FLOAT16(2), FLOAT16(4), FLOAT16(4), FLOAT16(4),
//        FLOAT16(4), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(4),
//        FLOAT16(3), FLOAT16(0), FLOAT16(2), FLOAT16(4),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(5),
//        FLOAT16(10), FLOAT16(2), FLOAT16(0), FLOAT16(10),
//        FLOAT16(3), FLOAT16(10), FLOAT16(1), FLOAT16(5),
//        FLOAT16(4), FLOAT16(0), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(3),
//        FLOAT16(10), FLOAT16(8), FLOAT16(6), FLOAT16(1),
//        FLOAT16(2), FLOAT16(5), FLOAT16(7), FLOAT16(5),
//        FLOAT16(4), FLOAT16(0), FLOAT16(6), FLOAT16(3),
//        FLOAT16(10), FLOAT16(9), FLOAT16(6), FLOAT16(9),
//        FLOAT16(1), FLOAT16(6), FLOAT16(5), FLOAT16(7),
//        FLOAT16(5), FLOAT16(2), FLOAT16(6), FLOAT16(6),
//        FLOAT16(1), FLOAT16(5), FLOAT16(6), FLOAT16(1),
//        FLOAT16(6), FLOAT16(4), FLOAT16(1), FLOAT16(6),
//        FLOAT16(2), FLOAT16(6), FLOAT16(5), FLOAT16(7),
//        FLOAT16(1), FLOAT16(9), FLOAT16(2), FLOAT16(6),
//        FLOAT16(6), FLOAT16(5), FLOAT16(10), FLOAT16(8),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 4, 4, 1), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d32843_i12843_a0) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_b;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 3, 2, 8, 4, 3 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 1, 2, 8, 4, 3 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(5), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(4), FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(7), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(8), FLOAT16(3), FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(5), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(0), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(3),
//        FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(10), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//        FLOAT16(9), FLOAT16(1), FLOAT16(0), FLOAT16(7), FLOAT16(9), FLOAT16(6), FLOAT16(8), FLOAT16(7),
//        FLOAT16(10), FLOAT16(9), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(5), FLOAT16(6), FLOAT16(9),
//        FLOAT16(4), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(1),
//        FLOAT16(1), FLOAT16(6), FLOAT16(8), FLOAT16(0), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(8),
//        FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(9), FLOAT16(1), FLOAT16(2), FLOAT16(7), FLOAT16(1),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(0), FLOAT16(7), FLOAT16(10), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(1), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(10), FLOAT16(6), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1), FLOAT16(6), FLOAT16(2), FLOAT16(5), FLOAT16(5), FLOAT16(10),
//        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(6), FLOAT16(1), FLOAT16(7), FLOAT16(6), FLOAT16(8),
//        FLOAT16(2), FLOAT16(5), FLOAT16(4), FLOAT16(2), FLOAT16(0), FLOAT16(9), FLOAT16(4), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1), FLOAT16(9), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(4),
//        FLOAT16(2), FLOAT16(1), FLOAT16(8), FLOAT16(5), FLOAT16(3), FLOAT16(4), FLOAT16(8), FLOAT16(10),
//        FLOAT16(7), FLOAT16(2), FLOAT16(7), FLOAT16(9), FLOAT16(2), FLOAT16(9), FLOAT16(5), FLOAT16(5),
//        FLOAT16(6), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(10), FLOAT16(6), FLOAT16(4), FLOAT16(9),
//        FLOAT16(7), FLOAT16(7), FLOAT16(10), FLOAT16(10), FLOAT16(9), FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(1), FLOAT16(4), FLOAT16(6), FLOAT16(9), FLOAT16(4), FLOAT16(8), FLOAT16(9), FLOAT16(7),
//        FLOAT16(8), FLOAT16(7), FLOAT16(8), FLOAT16(0), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(0),
//        FLOAT16(7), FLOAT16(5), FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(10), FLOAT16(9), FLOAT16(9),
//        FLOAT16(5), FLOAT16(1), FLOAT16(4), FLOAT16(10), FLOAT16(2), FLOAT16(4), FLOAT16(3), FLOAT16(5),
//        FLOAT16(9), FLOAT16(4), FLOAT16(5), FLOAT16(8), FLOAT16(4), FLOAT16(2), FLOAT16(10), FLOAT16(1),
//        FLOAT16(6), FLOAT16(6), FLOAT16(0), FLOAT16(0), FLOAT16(8), FLOAT16(8), FLOAT16(3), FLOAT16(4),
//        FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(9), FLOAT16(7), FLOAT16(9), FLOAT16(1), FLOAT16(0),
//        FLOAT16(8), FLOAT16(6), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(4), FLOAT16(10), FLOAT16(10),
//        FLOAT16(4), FLOAT16(2), FLOAT16(7), FLOAT16(3), FLOAT16(8), FLOAT16(8), FLOAT16(4), FLOAT16(3),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(10), FLOAT16(2), FLOAT16(9), FLOAT16(1), FLOAT16(4),
//        FLOAT16(6), FLOAT16(1), FLOAT16(9), FLOAT16(1), FLOAT16(10), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(6), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(6),
//        FLOAT16(0), FLOAT16(6), FLOAT16(2), FLOAT16(3), FLOAT16(7), FLOAT16(1), FLOAT16(8), FLOAT16(5),
//        FLOAT16(6), FLOAT16(6), FLOAT16(3), FLOAT16(7), FLOAT16(1), FLOAT16(1), FLOAT16(5), FLOAT16(9),
//        FLOAT16(8), FLOAT16(6), FLOAT16(8), FLOAT16(3), FLOAT16(1), FLOAT16(5), FLOAT16(3), FLOAT16(6),
//        FLOAT16(5), FLOAT16(4), FLOAT16(2), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(2), FLOAT16(7), FLOAT16(7), FLOAT16(5), FLOAT16(8),
//        FLOAT16(7), FLOAT16(10), FLOAT16(5), FLOAT16(10), FLOAT16(3), FLOAT16(5), FLOAT16(5), FLOAT16(7),
//        FLOAT16(4), FLOAT16(6), FLOAT16(10), FLOAT16(1), FLOAT16(7), FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(3), FLOAT16(7), FLOAT16(6), FLOAT16(10), FLOAT16(2), FLOAT16(10),
//        FLOAT16(2), FLOAT16(9), FLOAT16(7), FLOAT16(5), FLOAT16(8), FLOAT16(0), FLOAT16(1), FLOAT16(7),
//        FLOAT16(7), FLOAT16(4), FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(7), FLOAT16(3), FLOAT16(8),
//        FLOAT16(1), FLOAT16(0), FLOAT16(5), FLOAT16(0), FLOAT16(1), FLOAT16(9), FLOAT16(8), FLOAT16(8),
//        FLOAT16(4), FLOAT16(0), FLOAT16(6), FLOAT16(5), FLOAT16(0), FLOAT16(5), FLOAT16(4), FLOAT16(2),
//        FLOAT16(4), FLOAT16(6), FLOAT16(7), FLOAT16(7), FLOAT16(5), FLOAT16(3), FLOAT16(8), FLOAT16(4),
//        FLOAT16(7), FLOAT16(3), FLOAT16(0), FLOAT16(1), FLOAT16(5), FLOAT16(8), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(7), FLOAT16(3), FLOAT16(0), FLOAT16(5), FLOAT16(5), FLOAT16(5),
//        FLOAT16(4), FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7), FLOAT16(6), FLOAT16(7), FLOAT16(3),
//        FLOAT16(0), FLOAT16(10), FLOAT16(5), FLOAT16(0), FLOAT16(9), FLOAT16(0), FLOAT16(4), FLOAT16(5),
//        FLOAT16(6), FLOAT16(8), FLOAT16(7), FLOAT16(5), FLOAT16(0), FLOAT16(1), FLOAT16(10), FLOAT16(2),
//        FLOAT16(3), FLOAT16(6), FLOAT16(6), FLOAT16(1), FLOAT16(6), FLOAT16(10), FLOAT16(3), FLOAT16(9),
//        FLOAT16(10), FLOAT16(2), FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(9), FLOAT16(2), FLOAT16(8),
//        FLOAT16(7), FLOAT16(4), FLOAT16(2), FLOAT16(7), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(6),
//        FLOAT16(0), FLOAT16(1), FLOAT16(6), FLOAT16(4), FLOAT16(0), FLOAT16(7), FLOAT16(4), FLOAT16(9),
//        FLOAT16(1), FLOAT16(10), FLOAT16(0), FLOAT16(0), FLOAT16(5), FLOAT16(8), FLOAT16(10), FLOAT16(2),
//        FLOAT16(3), FLOAT16(8), FLOAT16(5), FLOAT16(8), FLOAT16(7), FLOAT16(7), FLOAT16(8), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(6), FLOAT16(7), FLOAT16(6), FLOAT16(4), FLOAT16(2), FLOAT16(2),
//        FLOAT16(7), FLOAT16(1), FLOAT16(8), FLOAT16(1), FLOAT16(0), FLOAT16(7), FLOAT16(1), FLOAT16(10),
//        FLOAT16(5), FLOAT16(6), FLOAT16(10), FLOAT16(0), FLOAT16(6), FLOAT16(7), FLOAT16(5), FLOAT16(0),
//        FLOAT16(4), FLOAT16(5), FLOAT16(8), FLOAT16(0), FLOAT16(4), FLOAT16(10), FLOAT16(5), FLOAT16(3),
//        FLOAT16(4), FLOAT16(8), FLOAT16(2), FLOAT16(1), FLOAT16(4), FLOAT16(10), FLOAT16(10), FLOAT16(2),
//        FLOAT16(0), FLOAT16(1), FLOAT16(5), FLOAT16(1), FLOAT16(5), FLOAT16(1), FLOAT16(9), FLOAT16(4),
//        FLOAT16(4), FLOAT16(3), FLOAT16(7), FLOAT16(6), FLOAT16(9), FLOAT16(8), FLOAT16(9), FLOAT16(7),
//        FLOAT16(4), FLOAT16(10), FLOAT16(6), FLOAT16(3), FLOAT16(5), FLOAT16(5), FLOAT16(4), FLOAT16(2),
//        FLOAT16(0), FLOAT16(4), FLOAT16(5), FLOAT16(3), FLOAT16(1), FLOAT16(2), FLOAT16(8), FLOAT16(5),
//        FLOAT16(7), FLOAT16(9), FLOAT16(2), FLOAT16(7), FLOAT16(2), FLOAT16(4), FLOAT16(0), FLOAT16(5),
//
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(8), FLOAT16(5), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(4), FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(9), FLOAT16(0), FLOAT16(5), FLOAT16(5),
//        FLOAT16(4), FLOAT16(4), FLOAT16(7), FLOAT16(9), FLOAT16(5), FLOAT16(8), FLOAT16(6), FLOAT16(4),
//        FLOAT16(8), FLOAT16(5), FLOAT16(8), FLOAT16(1), FLOAT16(4), FLOAT16(7), FLOAT16(5), FLOAT16(0),
//        FLOAT16(0), FLOAT16(5), FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(8), FLOAT16(5), FLOAT16(4),
//        FLOAT16(4), FLOAT16(1), FLOAT16(3), FLOAT16(9), FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(3),
//        FLOAT16(9), FLOAT16(10), FLOAT16(5), FLOAT16(0), FLOAT16(9), FLOAT16(3), FLOAT16(4), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(5), FLOAT16(0), FLOAT16(5), FLOAT16(3), FLOAT16(4),
//        FLOAT16(3), FLOAT16(6), FLOAT16(2), FLOAT16(9), FLOAT16(10), FLOAT16(10), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(2), FLOAT16(2), FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//        FLOAT16(4), FLOAT16(4), FLOAT16(7), FLOAT16(7), FLOAT16(9), FLOAT16(6), FLOAT16(4), FLOAT16(6),
//        FLOAT16(10), FLOAT16(9), FLOAT16(2), FLOAT16(10), FLOAT16(2), FLOAT16(7), FLOAT16(6), FLOAT16(9),
//        FLOAT16(1), FLOAT16(9), FLOAT16(2), FLOAT16(4), FLOAT16(10), FLOAT16(5), FLOAT16(3), FLOAT16(2),
//        FLOAT16(3), FLOAT16(6), FLOAT16(5), FLOAT16(0), FLOAT16(5), FLOAT16(8), FLOAT16(7), FLOAT16(8),
//        FLOAT16(0), FLOAT16(6), FLOAT16(2), FLOAT16(9), FLOAT16(6), FLOAT16(1), FLOAT16(7), FLOAT16(2),
//        FLOAT16(1), FLOAT16(3), FLOAT16(3), FLOAT16(7), FLOAT16(0), FLOAT16(7), FLOAT16(5), FLOAT16(9),
//        FLOAT16(8), FLOAT16(3), FLOAT16(10), FLOAT16(3), FLOAT16(1), FLOAT16(5), FLOAT16(4), FLOAT16(6),
//        FLOAT16(4), FLOAT16(5), FLOAT16(6), FLOAT16(4), FLOAT16(4), FLOAT16(10), FLOAT16(5), FLOAT16(1),
//        FLOAT16(3), FLOAT16(4), FLOAT16(2), FLOAT16(1), FLOAT16(7), FLOAT16(7), FLOAT16(5), FLOAT16(10),
//        FLOAT16(7), FLOAT16(1), FLOAT16(5), FLOAT16(10), FLOAT16(3), FLOAT16(1), FLOAT16(5), FLOAT16(4),
//        FLOAT16(2), FLOAT16(3), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(8), FLOAT16(5), FLOAT16(5),
//        FLOAT16(4), FLOAT16(4), FLOAT16(3), FLOAT16(3), FLOAT16(5), FLOAT16(10), FLOAT16(4), FLOAT16(2),
//        FLOAT16(2), FLOAT16(9), FLOAT16(7), FLOAT16(5), FLOAT16(3), FLOAT16(4), FLOAT16(8), FLOAT16(5),
//        FLOAT16(7), FLOAT16(4), FLOAT16(6), FLOAT16(8), FLOAT16(2), FLOAT16(7), FLOAT16(3), FLOAT16(5),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 8, 4, 3), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d223442_i226442_a5) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_x;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 2, 2, 3, 4, 4, 2 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 2, 2, 6, 4, 4, 2 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8),
//        FLOAT16(5), FLOAT16(5), FLOAT16(2),
//        FLOAT16(0), FLOAT16(7), FLOAT16(7),
//        FLOAT16(10), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(0),
//        FLOAT16(5), FLOAT16(7), FLOAT16(0),
//        FLOAT16(4), FLOAT16(0), FLOAT16(4),
//        FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1),
//        FLOAT16(7), FLOAT16(4), FLOAT16(7),
//        FLOAT16(10), FLOAT16(8), FLOAT16(2),
//        FLOAT16(0), FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10),
//        FLOAT16(4), FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8), FLOAT16(7),
//        FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8),
//        FLOAT16(5), FLOAT16(2), FLOAT16(3),
//        FLOAT16(3), FLOAT16(1), FLOAT16(5),
//        FLOAT16(9), FLOAT16(10), FLOAT16(0),
//        FLOAT16(9), FLOAT16(5), FLOAT16(5),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5),
//        FLOAT16(2), FLOAT16(0), FLOAT16(10),
//        FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5),
//        FLOAT16(5), FLOAT16(10), FLOAT16(0),
//        FLOAT16(8), FLOAT16(8), FLOAT16(9),
//        FLOAT16(1), FLOAT16(0), FLOAT16(7),
//        FLOAT16(9), FLOAT16(6), FLOAT16(8),
//        FLOAT16(7), FLOAT16(10), FLOAT16(9),
//        FLOAT16(2), FLOAT16(3), FLOAT16(3),
//        FLOAT16(5), FLOAT16(6), FLOAT16(9),
//        FLOAT16(4), FLOAT16(9), FLOAT16(2),
//        FLOAT16(4), FLOAT16(5), FLOAT16(5),
//        FLOAT16(3), FLOAT16(1), FLOAT16(1),
//        FLOAT16(6), FLOAT16(8), FLOAT16(0),
//        FLOAT16(5), FLOAT16(5), FLOAT16(10),
//        FLOAT16(8), FLOAT16(6), FLOAT16(9),
//        FLOAT16(6), FLOAT16(9), FLOAT16(1),
//        FLOAT16(2), FLOAT16(7), FLOAT16(1),
//        FLOAT16(1), FLOAT16(3), FLOAT16(0),
//        FLOAT16(4), FLOAT16(0), FLOAT16(7),
//        FLOAT16(10), FLOAT16(2), FLOAT16(1),
//        FLOAT16(3), FLOAT16(9), FLOAT16(7),
//        FLOAT16(1), FLOAT16(7), FLOAT16(4),
//        FLOAT16(4), FLOAT16(5), FLOAT16(1),
//        FLOAT16(6), FLOAT16(9), FLOAT16(6),
//        FLOAT16(10), FLOAT16(6), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1),
//        FLOAT16(6), FLOAT16(2), FLOAT16(5),
//        FLOAT16(5), FLOAT16(10), FLOAT16(1),
//        FLOAT16(2), FLOAT16(3), FLOAT16(6),
//        FLOAT16(1), FLOAT16(7), FLOAT16(6),
//        FLOAT16(8), FLOAT16(2), FLOAT16(5),
//        FLOAT16(4), FLOAT16(2), FLOAT16(0),
//        FLOAT16(9), FLOAT16(4), FLOAT16(1),
//        FLOAT16(10), FLOAT16(4), FLOAT16(1),
//        FLOAT16(9), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(4), FLOAT16(2),
//        FLOAT16(1), FLOAT16(8), FLOAT16(5),
//        FLOAT16(3), FLOAT16(4), FLOAT16(8),
//        FLOAT16(10), FLOAT16(7), FLOAT16(2),
//        FLOAT16(7), FLOAT16(9), FLOAT16(2),
//        FLOAT16(9), FLOAT16(5), FLOAT16(5),
//        FLOAT16(6), FLOAT16(8), FLOAT16(8),
//        FLOAT16(5), FLOAT16(10), FLOAT16(6),
//        FLOAT16(4), FLOAT16(9), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(10),
//        FLOAT16(9), FLOAT16(3), FLOAT16(5),
//        FLOAT16(5), FLOAT16(1), FLOAT16(4),
//        FLOAT16(6), FLOAT16(9), FLOAT16(4),
//        FLOAT16(8), FLOAT16(9), FLOAT16(7),
//        FLOAT16(8), FLOAT16(7), FLOAT16(8),
//        FLOAT16(0), FLOAT16(9), FLOAT16(5),
//        FLOAT16(5), FLOAT16(0), FLOAT16(7),
//        FLOAT16(5), FLOAT16(7), FLOAT16(7),
//        FLOAT16(2), FLOAT16(10), FLOAT16(9),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1),
//        FLOAT16(4), FLOAT16(10), FLOAT16(2),
//        FLOAT16(4), FLOAT16(3), FLOAT16(5),
//        FLOAT16(9), FLOAT16(4), FLOAT16(5),
//        FLOAT16(8), FLOAT16(4), FLOAT16(2),
//        FLOAT16(10), FLOAT16(1), FLOAT16(6),
//        FLOAT16(6), FLOAT16(0), FLOAT16(0),
//        FLOAT16(8), FLOAT16(8), FLOAT16(3),
//        FLOAT16(4), FLOAT16(7), FLOAT16(7),
//        FLOAT16(2), FLOAT16(9), FLOAT16(7),
//        FLOAT16(9), FLOAT16(1), FLOAT16(0),
//        FLOAT16(8), FLOAT16(6), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(4),
//        FLOAT16(10), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(7), FLOAT16(3),
//        FLOAT16(8), FLOAT16(8), FLOAT16(4),
//        FLOAT16(3), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(10), FLOAT16(2),
//        FLOAT16(9), FLOAT16(1), FLOAT16(4),
//        FLOAT16(6), FLOAT16(1), FLOAT16(9),
//        FLOAT16(1), FLOAT16(10), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(6), FLOAT16(7), FLOAT16(8),
//        FLOAT16(7), FLOAT16(8), FLOAT16(7),
//        FLOAT16(6), FLOAT16(0), FLOAT16(6),
//        FLOAT16(2), FLOAT16(3), FLOAT16(7),
//        FLOAT16(1), FLOAT16(8), FLOAT16(5),
//        FLOAT16(6), FLOAT16(6), FLOAT16(3),
//        FLOAT16(7), FLOAT16(1), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(8),
//        FLOAT16(6), FLOAT16(8), FLOAT16(3),
//        FLOAT16(1), FLOAT16(5), FLOAT16(3),
//        FLOAT16(6), FLOAT16(5), FLOAT16(4),
//        FLOAT16(2), FLOAT16(4), FLOAT16(4),
//        FLOAT16(4), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(0), FLOAT16(4),
//        FLOAT16(2), FLOAT16(7), FLOAT16(7),
//        FLOAT16(5), FLOAT16(8), FLOAT16(7),
//        FLOAT16(10), FLOAT16(5), FLOAT16(10),
//        FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(7), FLOAT16(4), FLOAT16(6),
//        FLOAT16(10), FLOAT16(1), FLOAT16(7),
//        FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(3),
//        FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(2), FLOAT16(10), FLOAT16(2),
//        FLOAT16(9), FLOAT16(7), FLOAT16(5),
//        FLOAT16(8), FLOAT16(0), FLOAT16(1),
//        FLOAT16(7), FLOAT16(7), FLOAT16(4),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10),
//        FLOAT16(7), FLOAT16(3), FLOAT16(8),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(1), FLOAT16(2),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(0),
//        FLOAT16(5), FLOAT16(5), FLOAT16(2), FLOAT16(5), FLOAT16(5), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(5), FLOAT16(4), FLOAT16(5), FLOAT16(4), FLOAT16(10), FLOAT16(5),
//        FLOAT16(0), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(9), FLOAT16(9),
//        FLOAT16(7), FLOAT16(0), FLOAT16(0), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(0), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(4),
//        FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(7), FLOAT16(7), FLOAT16(10),
//        FLOAT16(5), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(9),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(2), FLOAT16(10), FLOAT16(8), FLOAT16(8), FLOAT16(2), FLOAT16(2),
//        FLOAT16(8), FLOAT16(8), FLOAT16(0), FLOAT16(3), FLOAT16(0), FLOAT16(0),
//        FLOAT16(6), FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(8), FLOAT16(6),
//        FLOAT16(4), FLOAT16(10), FLOAT16(2), FLOAT16(10), FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(0), FLOAT16(6), FLOAT16(6), FLOAT16(9), FLOAT16(0), FLOAT16(0),
//        FLOAT16(8), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(4), FLOAT16(2),
//        FLOAT16(5), FLOAT16(3), FLOAT16(3), FLOAT16(5), FLOAT16(3), FLOAT16(5),
//        FLOAT16(3), FLOAT16(1), FLOAT16(1), FLOAT16(3), FLOAT16(1), FLOAT16(1),
//        FLOAT16(10), FLOAT16(9), FLOAT16(0), FLOAT16(10), FLOAT16(9), FLOAT16(0),
//        FLOAT16(9), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(5),
//        FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(3), FLOAT16(5), FLOAT16(10),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2), FLOAT16(0), FLOAT16(10), FLOAT16(10),
//        FLOAT16(0), FLOAT16(5), FLOAT16(4), FLOAT16(4), FLOAT16(5), FLOAT16(0),
//        FLOAT16(10), FLOAT16(3), FLOAT16(5), FLOAT16(5), FLOAT16(10), FLOAT16(10),
//        FLOAT16(10), FLOAT16(5), FLOAT16(10), FLOAT16(0), FLOAT16(10), FLOAT16(10),
//        FLOAT16(8), FLOAT16(9), FLOAT16(8), FLOAT16(9), FLOAT16(8), FLOAT16(9),
//        FLOAT16(7), FLOAT16(0), FLOAT16(0), FLOAT16(7), FLOAT16(0), FLOAT16(0),
//        FLOAT16(8), FLOAT16(9), FLOAT16(6), FLOAT16(8), FLOAT16(8), FLOAT16(6),
//        FLOAT16(9), FLOAT16(9), FLOAT16(7), FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(2), FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(2), FLOAT16(3),
//        FLOAT16(6), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(6), FLOAT16(5),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(9), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(5), FLOAT16(4), FLOAT16(4), FLOAT16(5), FLOAT16(5),
//        FLOAT16(3), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(8), FLOAT16(8), FLOAT16(6), FLOAT16(6), FLOAT16(0),
//        FLOAT16(10), FLOAT16(10), FLOAT16(5), FLOAT16(10), FLOAT16(5), FLOAT16(5),
//        FLOAT16(6), FLOAT16(8), FLOAT16(9), FLOAT16(9), FLOAT16(8), FLOAT16(9),
//        FLOAT16(9), FLOAT16(6), FLOAT16(6), FLOAT16(1), FLOAT16(9), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(1), FLOAT16(7), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(3), FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(0), FLOAT16(4), FLOAT16(4), FLOAT16(7), FLOAT16(4), FLOAT16(0),
//        FLOAT16(10), FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(10), FLOAT16(10),
//        FLOAT16(3), FLOAT16(3), FLOAT16(3), FLOAT16(9), FLOAT16(9), FLOAT16(7),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(1), FLOAT16(4), FLOAT16(4),
//        FLOAT16(4), FLOAT16(4), FLOAT16(1), FLOAT16(1), FLOAT16(5), FLOAT16(5),
//        FLOAT16(9), FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(9), FLOAT16(9),
//        FLOAT16(6), FLOAT16(10), FLOAT16(6), FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(1), FLOAT16(10), FLOAT16(1), FLOAT16(10), FLOAT16(1), FLOAT16(10),
//        FLOAT16(2), FLOAT16(5), FLOAT16(6), FLOAT16(2), FLOAT16(2), FLOAT16(6),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(1), FLOAT16(10), FLOAT16(10),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(6),
//        FLOAT16(1), FLOAT16(1), FLOAT16(6), FLOAT16(7), FLOAT16(7), FLOAT16(6),
//        FLOAT16(8), FLOAT16(5), FLOAT16(5), FLOAT16(8), FLOAT16(8), FLOAT16(2),
//        FLOAT16(4), FLOAT16(2), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(4),
//        FLOAT16(1), FLOAT16(9), FLOAT16(4), FLOAT16(9), FLOAT16(9), FLOAT16(4),
//        FLOAT16(1), FLOAT16(4), FLOAT16(1), FLOAT16(4), FLOAT16(4), FLOAT16(10),
//        FLOAT16(1), FLOAT16(1), FLOAT16(9), FLOAT16(1), FLOAT16(9), FLOAT16(1),
//        FLOAT16(4), FLOAT16(4), FLOAT16(0), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(8), FLOAT16(1), FLOAT16(1), FLOAT16(1), FLOAT16(5), FLOAT16(8),
//        FLOAT16(3), FLOAT16(4), FLOAT16(3), FLOAT16(3), FLOAT16(3), FLOAT16(8),
//        FLOAT16(10), FLOAT16(10), FLOAT16(7), FLOAT16(10), FLOAT16(10), FLOAT16(2),
//        FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(9), FLOAT16(9), FLOAT16(9),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(9), FLOAT16(9), FLOAT16(9),
//        FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(8),
//        FLOAT16(5), FLOAT16(6), FLOAT16(6), FLOAT16(5), FLOAT16(10), FLOAT16(5),
//        FLOAT16(7), FLOAT16(9), FLOAT16(7), FLOAT16(7), FLOAT16(9), FLOAT16(7),
//        FLOAT16(10), FLOAT16(10), FLOAT16(7), FLOAT16(10), FLOAT16(7), FLOAT16(10),
//        FLOAT16(5), FLOAT16(3), FLOAT16(9), FLOAT16(3), FLOAT16(9), FLOAT16(3),
//        FLOAT16(5), FLOAT16(1), FLOAT16(1), FLOAT16(4), FLOAT16(4), FLOAT16(4),
//        FLOAT16(9), FLOAT16(9), FLOAT16(9), FLOAT16(4), FLOAT16(6), FLOAT16(6),
//        FLOAT16(9), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(7), FLOAT16(9),
//        FLOAT16(8), FLOAT16(8), FLOAT16(7), FLOAT16(8), FLOAT16(8), FLOAT16(8),
//        FLOAT16(9), FLOAT16(0), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(5), FLOAT16(7), FLOAT16(7), FLOAT16(0), FLOAT16(0),
//        FLOAT16(5), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(2), FLOAT16(9), FLOAT16(2), FLOAT16(9), FLOAT16(9), FLOAT16(10),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(1), FLOAT16(5), FLOAT16(9),
//        FLOAT16(4), FLOAT16(10), FLOAT16(2), FLOAT16(10), FLOAT16(4), FLOAT16(4),
//        FLOAT16(5), FLOAT16(3), FLOAT16(4), FLOAT16(3), FLOAT16(4), FLOAT16(5),
//        FLOAT16(5), FLOAT16(9), FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(4),
//        FLOAT16(4), FLOAT16(8), FLOAT16(8), FLOAT16(2), FLOAT16(4), FLOAT16(4),
//        FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(1), FLOAT16(10), FLOAT16(6),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(6), FLOAT16(0), FLOAT16(0),
//        FLOAT16(3), FLOAT16(8), FLOAT16(8), FLOAT16(3), FLOAT16(8), FLOAT16(8),
//        FLOAT16(4), FLOAT16(7), FLOAT16(4), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(9), FLOAT16(2), FLOAT16(7), FLOAT16(9), FLOAT16(7), FLOAT16(7),
//        FLOAT16(9), FLOAT16(0), FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(2), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(0), FLOAT16(2), FLOAT16(0),
//        FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(3), FLOAT16(7), FLOAT16(3),
//        FLOAT16(4), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(8), FLOAT16(8),
//        FLOAT16(3), FLOAT16(0), FLOAT16(3), FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(2), FLOAT16(10), FLOAT16(10), FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(9), FLOAT16(4), FLOAT16(1), FLOAT16(1), FLOAT16(4), FLOAT16(4),
//        FLOAT16(6), FLOAT16(1), FLOAT16(6), FLOAT16(9), FLOAT16(6), FLOAT16(1),
//        FLOAT16(10), FLOAT16(2), FLOAT16(1), FLOAT16(10), FLOAT16(1), FLOAT16(10),
//        FLOAT16(2), FLOAT16(1), FLOAT16(1), FLOAT16(2), FLOAT16(2), FLOAT16(1),
//        FLOAT16(8), FLOAT16(6), FLOAT16(6), FLOAT16(8), FLOAT16(6), FLOAT16(6),
//        FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(0), FLOAT16(0), FLOAT16(6), FLOAT16(6), FLOAT16(0), FLOAT16(0),
//        FLOAT16(7), FLOAT16(3), FLOAT16(3), FLOAT16(2), FLOAT16(7), FLOAT16(3),
//        FLOAT16(5), FLOAT16(1), FLOAT16(1), FLOAT16(5), FLOAT16(8), FLOAT16(5),
//        FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(6),
//        FLOAT16(1), FLOAT16(1), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(7),
//        FLOAT16(9), FLOAT16(5), FLOAT16(8), FLOAT16(8), FLOAT16(5), FLOAT16(9),
//        FLOAT16(6), FLOAT16(8), FLOAT16(8), FLOAT16(6), FLOAT16(6), FLOAT16(6),
//        FLOAT16(3), FLOAT16(5), FLOAT16(3), FLOAT16(5), FLOAT16(1), FLOAT16(1),
//        FLOAT16(6), FLOAT16(5), FLOAT16(4), FLOAT16(5), FLOAT16(6), FLOAT16(5),
//        FLOAT16(4), FLOAT16(2), FLOAT16(4), FLOAT16(4), FLOAT16(2), FLOAT16(2),
//        FLOAT16(4), FLOAT16(5), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(4),
//        FLOAT16(3), FLOAT16(3), FLOAT16(0), FLOAT16(4), FLOAT16(3), FLOAT16(4),
//        FLOAT16(7), FLOAT16(7), FLOAT16(2), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(5), FLOAT16(7), FLOAT16(8), FLOAT16(7), FLOAT16(5), FLOAT16(5),
//        FLOAT16(10), FLOAT16(5), FLOAT16(10), FLOAT16(10), FLOAT16(10), FLOAT16(5),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(5), FLOAT16(5),
//        FLOAT16(6), FLOAT16(6), FLOAT16(7), FLOAT16(7), FLOAT16(7), FLOAT16(7),
//        FLOAT16(10), FLOAT16(1), FLOAT16(7), FLOAT16(1), FLOAT16(7), FLOAT16(7),
//        FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(5), FLOAT16(3), FLOAT16(5),
//        FLOAT16(0), FLOAT16(9), FLOAT16(3), FLOAT16(9), FLOAT16(0), FLOAT16(3),
//        FLOAT16(6), FLOAT16(6), FLOAT16(6), FLOAT16(10), FLOAT16(10), FLOAT16(6),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2), FLOAT16(10), FLOAT16(10), FLOAT16(10),
//        FLOAT16(5), FLOAT16(9), FLOAT16(7), FLOAT16(7), FLOAT16(5), FLOAT16(9),
//        FLOAT16(0), FLOAT16(8), FLOAT16(0), FLOAT16(1), FLOAT16(1), FLOAT16(8),
//        FLOAT16(7), FLOAT16(7), FLOAT16(4), FLOAT16(4), FLOAT16(4), FLOAT16(4),
//        FLOAT16(8), FLOAT16(10), FLOAT16(8), FLOAT16(6), FLOAT16(10), FLOAT16(8),
//        FLOAT16(3), FLOAT16(3), FLOAT16(7), FLOAT16(8), FLOAT16(3), FLOAT16(8),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(2, 2, 6, 4, 4, 2), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d124251_i124221_an3) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_z;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 1, 2, 4, 2, 5, 1 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 1, 2, 4, 2, 2, 1 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8), FLOAT16(5),
//        FLOAT16(5), FLOAT16(2), FLOAT16(0), FLOAT16(7),
//        FLOAT16(7), FLOAT16(10), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(0), FLOAT16(5),
//        FLOAT16(7), FLOAT16(0), FLOAT16(4), FLOAT16(0),
//        FLOAT16(4), FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1), FLOAT16(7),
//        FLOAT16(4), FLOAT16(7), FLOAT16(10), FLOAT16(8),
//        FLOAT16(2), FLOAT16(0), FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10), FLOAT16(4),
//        FLOAT16(2), FLOAT16(10), FLOAT16(7), FLOAT16(8),
//        FLOAT16(7), FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8), FLOAT16(5),
//        FLOAT16(2), FLOAT16(3), FLOAT16(3), FLOAT16(1),
//        FLOAT16(5), FLOAT16(9), FLOAT16(10), FLOAT16(0),
//        FLOAT16(9), FLOAT16(5), FLOAT16(5), FLOAT16(3),
//        FLOAT16(10), FLOAT16(5), FLOAT16(2), FLOAT16(0),
//        FLOAT16(10), FLOAT16(0), FLOAT16(5), FLOAT16(4),
//        FLOAT16(3), FLOAT16(10), FLOAT16(5), FLOAT16(5),
//        FLOAT16(10), FLOAT16(0), FLOAT16(8), FLOAT16(8),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(2), FLOAT16(4), FLOAT16(3),
//        FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(4), FLOAT16(0), FLOAT16(1), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
//        FLOAT16(3), FLOAT16(1), FLOAT16(4), FLOAT16(2),
//        FLOAT16(4), FLOAT16(2), FLOAT16(1), FLOAT16(3),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(4),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(4),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(0), FLOAT16(8), FLOAT16(7),
//        FLOAT16(6), FLOAT16(2), FLOAT16(0), FLOAT16(5),
//        FLOAT16(2), FLOAT16(1), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(2), FLOAT16(0), FLOAT16(5),
//        FLOAT16(10), FLOAT16(4), FLOAT16(5), FLOAT16(0),
//        FLOAT16(10), FLOAT16(5), FLOAT16(3), FLOAT16(4),
//        FLOAT16(5), FLOAT16(4), FLOAT16(10), FLOAT16(5),
//        FLOAT16(2), FLOAT16(0), FLOAT16(5), FLOAT16(8),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 4, 2, 2, 1), axis);
//}
//
//TEST(gather_elements_gpu_fp16, d233113_i233115_a2) {
//    auto& engine = get_test_engine();
//
//    auto axis = cldnn::gather_elements::gather_elements_axis::along_w;
//    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 2, 3, 3, 1, 1, 3 } }); // data
//    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 2, 3, 3, 1, 1, 5 } }); // indices
//
//    set_values(input0, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(8),
//        FLOAT16(5), FLOAT16(5), FLOAT16(2),
//        FLOAT16(0), FLOAT16(7), FLOAT16(7),
//        FLOAT16(10), FLOAT16(4), FLOAT16(5),
//        FLOAT16(9), FLOAT16(0), FLOAT16(0),
//        FLOAT16(5), FLOAT16(7), FLOAT16(0),
//        FLOAT16(4), FLOAT16(0), FLOAT16(4),
//        FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1),
//        FLOAT16(7), FLOAT16(4), FLOAT16(7),
//        FLOAT16(10), FLOAT16(8), FLOAT16(2),
//        FLOAT16(0), FLOAT16(8), FLOAT16(3),
//        FLOAT16(6), FLOAT16(8), FLOAT16(10),
//        FLOAT16(4), FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8), FLOAT16(7),
//        FLOAT16(0), FLOAT16(6), FLOAT16(9),
//        FLOAT16(2), FLOAT16(4), FLOAT16(8),
//        FLOAT16(5), FLOAT16(2), FLOAT16(3),
//    });
//
//    set_values(input1, {
//        FLOAT16(0), FLOAT16(1), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(1),
//        FLOAT16(1), FLOAT16(0), FLOAT16(2),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(2), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(0), FLOAT16(2),
//        FLOAT16(2), FLOAT16(0), FLOAT16(1),
//        FLOAT16(1), FLOAT16(2), FLOAT16(2),
//        FLOAT16(1), FLOAT16(1), FLOAT16(0),
//        FLOAT16(2), FLOAT16(0), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(2),
//        FLOAT16(2), FLOAT16(1), FLOAT16(0),
//        FLOAT16(0), FLOAT16(2), FLOAT16(1),
//        FLOAT16(2), FLOAT16(1), FLOAT16(2),
//        FLOAT16(0), FLOAT16(0), FLOAT16(1),
//        FLOAT16(2), FLOAT16(0), FLOAT16(2),
//    });
//
//    std::vector<float> expected_results = {
//        FLOAT16(0), FLOAT16(5), FLOAT16(7),
//        FLOAT16(0), FLOAT16(7), FLOAT16(8),
//        FLOAT16(0), FLOAT16(1), FLOAT16(7),
//        FLOAT16(0), FLOAT16(1), FLOAT16(8),
//        FLOAT16(5), FLOAT16(1), FLOAT16(2),
//        FLOAT16(9), FLOAT16(7), FLOAT16(0),
//        FLOAT16(5), FLOAT16(0), FLOAT16(0),
//        FLOAT16(9), FLOAT16(4), FLOAT16(0),
//        FLOAT16(9), FLOAT16(4), FLOAT16(0),
//        FLOAT16(5), FLOAT16(4), FLOAT16(5),
//        FLOAT16(7), FLOAT16(5), FLOAT16(1),
//        FLOAT16(7), FLOAT16(6), FLOAT16(10),
//        FLOAT16(7), FLOAT16(0), FLOAT16(1),
//        FLOAT16(4), FLOAT16(5), FLOAT16(1),
//        FLOAT16(9), FLOAT16(5), FLOAT16(1),
//        FLOAT16(7), FLOAT16(4), FLOAT16(3),
//        FLOAT16(10), FLOAT16(8), FLOAT16(3),
//        FLOAT16(0), FLOAT16(8), FLOAT16(7),
//        FLOAT16(0), FLOAT16(4), FLOAT16(7),
//        FLOAT16(7), FLOAT16(4), FLOAT16(3),
//        FLOAT16(7), FLOAT16(8), FLOAT16(10),
//        FLOAT16(4), FLOAT16(8), FLOAT16(7),
//        FLOAT16(4), FLOAT16(2), FLOAT16(10),
//        FLOAT16(7), FLOAT16(8), FLOAT16(10),
//        FLOAT16(6), FLOAT16(8), FLOAT16(7),
//        FLOAT16(5), FLOAT16(4), FLOAT16(9),
//        FLOAT16(0), FLOAT16(2), FLOAT16(8),
//        FLOAT16(5), FLOAT16(4), FLOAT16(3),
//        FLOAT16(0), FLOAT16(6), FLOAT16(8),
//        FLOAT16(5), FLOAT16(6), FLOAT16(3),
//    };
//
//    DoTest(engine, input0, input1, expected_results, tensor(2, 3, 3, 1, 1, 5), axis);
//}
