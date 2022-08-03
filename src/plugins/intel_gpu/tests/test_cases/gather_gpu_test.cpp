// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include <cstddef>
#include <array>

using namespace cldnn;
using namespace ::tests;

template <class T>
int get_not_one_dim(const T& a) {
    int ret = a.size();
    while (ret - 1 >= 0 && a[ret - 1] == 1)
        ret--;
    return ret;
};

template <class T>
int get_linear_size(const T& a) {
    return std::accumulate(a.begin(), a.end(), 1, [](int x, int y) {
        return x * y;
    });
};

using gather8_test_param = std::tuple<int,                // batch_dim, value in [0,get_not_one_dim(dict))
                                      int,                // axis, value in [batch_dim,get_not_one_dim(dict))
                                      format::type,       // format of input0
                                      format::type,       // format of input1
                                      std::vector<int>,   // shape of input0, order = default_format
                                      std::vector<int>>;  // shape of input1, order = default_format
template <class T_dat, class T_ind, data_types T_dat_dt, data_types T_ind_dt>
class gather8_test : public ::testing::TestWithParam<gather8_test_param> {
public:
    int axis, batch_dim;
    std::array<format::type, 3> fmt;
    std::vector<int> shape_in[2];
    std::vector<int> shape_out;

    void SetUp() override {
        auto& engine = get_test_engine();

        std::tie(batch_dim, axis, fmt[0], fmt[1], shape_in[0], shape_in[1]) = GetParam();
        fmt[2] = fmt[0];

        // refer: src/core/shape_inference/include/gather_shape_inference.hpp
        size_t out_dim = get_not_one_dim(shape_in[0]) - 1 + get_not_one_dim(shape_in[1]) - batch_dim;
        shape_out = std::vector<int>(std::max(shape_in[0].size(), out_dim), 1);
        for (int i = 0; i < batch_dim; i++)  // batch_dim
            shape_out[i] = shape_in[0][i];
        for (int i = batch_dim; i < axis; i++)  // before axis = shape_in[0][..]
            shape_out[i] = shape_in[0][i];
        for (int i = batch_dim; i < get_not_one_dim(shape_in[1]); i++)  // axis = shape_in[1]
            shape_out[axis + (i - batch_dim)] = shape_in[1][i];
        for (int i = axis + 1; i < get_not_one_dim(shape_in[0]); i++)  // after axis = shape_in[0][..]
            shape_out[axis + get_not_one_dim(shape_in[1]) - batch_dim + (i - axis - 1)] = shape_in[0][i];

        auto dat = generate_random_1d<T_dat>(get_linear_size(shape_in[0]), -99, 99);
        auto input0 =
            engine.allocate_memory(layout(T_dat_dt,
                                          format::get_default_format(shape_in[0].size()),
                                          tensor(format::get_default_format(shape_in[0].size()), shape_in[0])));
        set_values(input0, dat);

        auto ind =
            generate_random_1d<T_ind>(get_linear_size(shape_in[1]), -shape_in[0][axis], shape_in[0][axis] - 1, 1);
        auto input1 =
            engine.allocate_memory(layout(T_ind_dt,
                                          format::get_default_format(shape_in[1].size()),
                                          tensor(format::get_default_format(shape_in[1].size()), shape_in[1])));
        set_values(input1, ind);

        topology reorder_topo;
        reorder_topo.add(input_layout("input0", input0->get_layout()));
        reorder_topo.add(input_layout("input1", input1->get_layout()));
        reorder_topo.add(reorder("reorder0", "input0", fmt[0], T_dat_dt));
        reorder_topo.add(reorder("reorder1", "input1", fmt[1], T_ind_dt));
        reorder_topo.add(gather("gather",
                                "reorder0",
                                "reorder1",
                                axis,
                                ov::Shape(shape_out.begin(), shape_out.end()),
                                batch_dim,
                                true));
        reorder_topo.add(reorder("reorder2", "gather", format::type::bfwzyx, T_dat_dt));
        network reorder_network(engine, reorder_topo);
        reorder_network.set_input_data("input0", input0);
        reorder_network.set_input_data("input1", input1);
        auto reorder_output = reorder_network.execute().at("reorder2").get_memory();
        cldnn::mem_lock<T_dat> reorder_output_ptr(reorder_output, get_test_stream());

        topology planar_topo;
        planar_topo.add(input_layout("input0", input0->get_layout()));
        planar_topo.add(input_layout("input1", input1->get_layout()));
        planar_topo.add(
            gather("gather", "input0", "input1", axis, ov::Shape(shape_out.begin(), shape_out.end()), batch_dim, true));
        network planar_network(engine, planar_topo);
        planar_network.set_input_data("input0", input0);
        planar_network.set_input_data("input1", input1);
        auto planar_output = planar_network.execute().at("gather").get_memory();
        cldnn::mem_lock<T_dat> planar_output_ptr(planar_output, get_test_stream());

        EXPECT_TRUE(
            !memcmp(reorder_output_ptr.data(), planar_output_ptr.data(), get_linear_size(shape_out) * sizeof(T_dat)));
    }
};
using gather8_test_f16i32 = gather8_test<FLOAT16, int, data_types::f16, data_types::i32>;
using gather8_test_f32i8 = gather8_test<float, char, data_types::f32, data_types::i8>;
using gather8_test_i32i32 = gather8_test<int, int, data_types::i32, data_types::i32>;
TEST_P(gather8_test_f16i32, gather8_test_f16i32) {}
TEST_P(gather8_test_f32i8, gather8_test_f32i8) {}
TEST_P(gather8_test_i32i32, gather8_test_i32i32) {}

// Important testcases
INSTANTIATE_TEST_SUITE_P(gather8_bd0_d4_i1,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),  // bdim in [0,get_not_one_dim(dict))
                                          testing::Values(0),  // axis in [batch_dim,get_not_one_dim(dict))
                                          testing::Values(format::type::bfyx),
                                          testing::Values(format::type::bfyx),
                                          testing::Values(std::vector<int>{5, 44, 7, 8}),
                                          testing::Values(std::vector<int>{4, 1, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bd0_d2_i2,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(1),
                                          testing::Values(format::type::b_fs_yx_fsv4),
                                          testing::Values(format::type::b_fs_yx_fsv16),
                                          testing::Values(std::vector<int>{8, 67, 1, 1}),
                                          testing::Values(std::vector<int>{4, 56, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_fs_b_yx_fsv32,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(0, 1, 2),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(std::vector<int>{3, 77, 4, 1}),
                                          testing::Values(std::vector<int>{2, 66, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_fs_b_yx_fsv32_bd1,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(1),
                                          testing::Values(1, 2),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(std::vector<int>{3, 77, 44, 1}),
                                          testing::Values(std::vector<int>{3, 66, 55, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_fs_b_yx_fsv32_bd2,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(2),
                                          testing::Values(2, 3),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(std::vector<int>{3, 4, 44, 6}),
                                          testing::Values(std::vector<int>{3, 4, 5, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bs_fs_yx_bsv16_fsv16_bd0_dim4_to_dim5,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(0, 2),
                                          testing::Values(format::type::bs_fs_yx_bsv16_fsv16),
                                          testing::Values(format::type::b_fs_yx_fsv32),
                                          testing::Values(std::vector<int>{3, 77, 44, 1}),
                                          testing::Values(std::vector<int>{3, 66, 55, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_b_fs_yx_fsv16_bd0_dim4_to_dim5,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(0, 2),
                                          testing::Values(format::type::b_fs_yx_fsv16),
                                          testing::Values(format::type::b_fs_yx_fsv4),
                                          testing::Values(std::vector<int>{3, 77, 44, 1}),
                                          testing::Values(std::vector<int>{3, 66, 55, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bfyx_bd0_dim4_to_dim6,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(0, 2),
                                          testing::Values(format::type::bfyx),
                                          testing::Values(format::type::b_fs_yx_fsv4),
                                          testing::Values(std::vector<int>{3, 7, 4, 6}),
                                          testing::Values(std::vector<int>{3, 6, 5, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bd0_d4_i1,
                         gather8_test_f32i8,
                         testing::Combine(testing::Values(0),
                                          testing::Values(2),
                                          testing::Values(format::type::bs_fs_yx_bsv16_fsv16),
                                          testing::Values(format::type::bs_fs_yx_bsv32_fsv16),
                                          testing::Values(std::vector<int>{5, 44, 7, 8}),
                                          testing::Values(std::vector<int>{4, 1, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bd0_d3_i3,
                         gather8_test_f32i8,
                         testing::Combine(testing::Values(0),
                                          testing::Values(1),
                                          testing::Values(format::type::b_fs_zyx_fsv16),
                                          testing::Values(format::type::bfzyx),
                                          testing::Values(std::vector<int>{8, 67, 3, 1, 1}),
                                          testing::Values(std::vector<int>{3, 56, 9, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_b_fs_zyx_fsv32,
                         gather8_test_f32i8,
                         testing::Combine(testing::Values(1),
                                          testing::Values(2),
                                          testing::Values(format::type::b_fs_zyx_fsv32),
                                          testing::Values(format::type::b_fs_zyx_fsv16),
                                          testing::Values(std::vector<int>{8, 66, 3, 1, 1}),
                                          testing::Values(std::vector<int>{8, 56, 9, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_b_fs_yx_fsv4,
                         gather8_test_i32i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(2),
                                          testing::Values(format::type::b_fs_yx_fsv4),
                                          testing::Values(format::type::b_fs_yx_fsv4),
                                          testing::Values(std::vector<int>{4, 6, 2, 1}),
                                          testing::Values(std::vector<int>{3, 5, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_bfyx,
                         gather8_test_i32i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(1),
                                          testing::Values(format::type::bfyx),
                                          testing::Values(format::type::bfyx),
                                          testing::Values(std::vector<int>{4, 3, 2, 1}),
                                          testing::Values(std::vector<int>{5, 6, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_byxf,
                         gather8_test_i32i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(2),
                                          testing::Values(format::type::byxf),
                                          testing::Values(format::type::byxf),
                                          testing::Values(std::vector<int>{4, 6, 2, 1}),
                                          testing::Values(std::vector<int>{3, 5, 1, 1})));
INSTANTIATE_TEST_SUITE_P(gather8_fs_b_yx_fsv32_2,
                         gather8_test_i32i32,
                         testing::Combine(testing::Values(0),
                                          testing::Values(2),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(format::type::fs_b_yx_fsv32),
                                          testing::Values(std::vector<int>{4, 6, 2, 3}),
                                          testing::Values(std::vector<int>{3, 1, 1, 1})));

// Disabled tests for faster CI
// Remove DISABLED_ prefix to test these cases
#define FORMAT4D                                                                                             \
    format::type::bfyx, format::type::b_fs_yx_fsv16, format::type::bs_fs_yx_bsv16_fsv16, format::type::byxf, \
        format::type::yxfb, format::type::fyxb, format::type::fs_b_yx_fsv32
#define FORMAT5D format::type::bfzyx, format::type::b_fs_zyx_fsv16, format::type::bs_fs_zyx_bsv16_fsv16
#define FORMAT6D format::type::bfwzyx
INSTANTIATE_TEST_SUITE_P(DISABLED_gather8_4d_f16i32,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0, 1),
                                          testing::Values(1, 2),
                                          testing::Values(FORMAT4D),
                                          testing::Values(FORMAT4D),
                                          testing::Values(std::vector<int>{3, 6, 2, 1}),
                                          testing::Values(std::vector<int>{3, 5, 1, 1})));
INSTANTIATE_TEST_SUITE_P(DISABLED_gather8_5d_f16i32,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0, 1),
                                          testing::Values(1, 2, 3),
                                          testing::Values(FORMAT5D),
                                          testing::Values(FORMAT5D),
                                          testing::Values(std::vector<int>{3, 6, 2, 7, 1}),
                                          testing::Values(std::vector<int>{3, 5, 1, 1, 1})));
INSTANTIATE_TEST_SUITE_P(DISABLED_gather8_6d_f16i32,
                         gather8_test_f16i32,
                         testing::Combine(testing::Values(0, 1),
                                          testing::Values(1, 2, 3),
                                          testing::Values(FORMAT6D),
                                          testing::Values(FORMAT6D),
                                          testing::Values(std::vector<int>{3, 6, 2, 7, 1, 1}),
                                          testing::Values(std::vector<int>{3, 5, 4, 1, 1, 1})));

TEST(gather8_gpu_fp16, d323_axisY_bdim_m1) {
    //  Dictionary : 3x2x3x4x2
    //  Indexes : 3x2x3x1
    //  Axis : 3
    //  batch_dim : -1
    //  Output : 3x2x3x3x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 0.f, 0.f, 3.f, -3.f, 0.f, 1.f, -3.f, 1.f, -2.f, 0.f, 3.f, -1.f, 1.f, 0.f, 2.f, 0.f, 1.f
    //
    //  Dictionary:
    //  1.f   2.f   3.f   4.f   5.f   6.f   7.f   8.f   9.f   10.f  11.f  12.f  13.f  14.f  15.f  16.f  17.f  18.f
    //  19.f  20.f  21.f  22.f  23.f  24.f  25.f  26.f  27.f  28.f  29.f  30.f  31.f  32.f  33.f  34.f  35.f  36.f
    //  37.f  38.f  39.f  40.f  41.f  42.f  43.f  44.f  45.f  46.f  47.f  48.f  49.f  50.f  51.f  52.f  53.f  54.f
    //  55.f  56.f  57.f  58.f  59.f  60.f  61.f  62.f  63.f  64.f  65.f  66.f  67.f  68.f  69.f  70.f  71.f  72.f
    //  73.f  74.f  75.f  76.f  77.f  78.f  79.f  80.f  81.f  82.f  83.f  84.f  85.f  86.f  87.f  88.f  89.f  90.f
    //  91.f  92.f  93.f  94.f  95.f  96.f  97.f  98.f  99.f  100.f 101.f 102.f 103.f 104.f 105.f 106.f 107.f 108.f
    //  109.f 110.f 111.f 112.f 113.f 114.f 115.f 116.f 117.f 118.f 119.f 120.f 121.f 122.f 123.f 124.f 125.f 126.f
    //  127.f 128.f 129.f 130.f 131.f 132.f 133.f 134.f 135.f 136.f 137.f 138.f 139.f 140.f 141.f 142.f 143.f 144.f
    //
    //  Output:
    //  1.f,    2.f,    9.f,    10.f,   17.f,   18.f,   31.f,   32.f,
    //  35.f,   36.f,   41.f,   42.f,   51.f,   52.f,   59.f,   60.f,
    //  67.f,   68.f,   77.f,   78.f,   81.f,   82.f,   95.f,   96.f,
    //  103.f,  104.f,  107.f,  108.f,  113.f,  114.f,  125.f,  126.f,
    //  129.f,  130.f,  139.f,  140.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 3, 2, 2, 4, 3} }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 3 } }); // Indexes
    int64_t axis = 3;
    int64_t batch_dim = -1;
    bool negative_indexes = true;

    set_values(input1, {
        FLOAT16(1.f),   FLOAT16(2.f),   FLOAT16(3.f),   FLOAT16(4.f),   FLOAT16(5.f),   FLOAT16(6.f),   FLOAT16(7.f),   FLOAT16(8.f),
        FLOAT16(9.f),   FLOAT16(10.f),  FLOAT16(11.f),  FLOAT16(12.f),  FLOAT16(13.f),  FLOAT16(14.f),  FLOAT16(15.f),  FLOAT16(16.f),
        FLOAT16(17.f),  FLOAT16(18.f),  FLOAT16(19.f),  FLOAT16(20.f),  FLOAT16(21.f),  FLOAT16(22.f),  FLOAT16(23.f),  FLOAT16(24.f),

        FLOAT16(25.f),  FLOAT16(26.f),  FLOAT16(27.f),  FLOAT16(28.f),  FLOAT16(29.f),  FLOAT16(30.f),  FLOAT16(31.f),  FLOAT16(32.f),
        FLOAT16(33.f),  FLOAT16(34.f),  FLOAT16(35.f),  FLOAT16(36.f),  FLOAT16(37.f),  FLOAT16(38.f),  FLOAT16(39.f),  FLOAT16(40.f),
        FLOAT16(41.f),  FLOAT16(42.f),  FLOAT16(43.f),  FLOAT16(44.f),  FLOAT16(45.f),  FLOAT16(46.f),  FLOAT16(47.f),  FLOAT16(48.f),


        FLOAT16(49.f),  FLOAT16(50.f),  FLOAT16(51.f),  FLOAT16(52.f),  FLOAT16(53.f),  FLOAT16(54.f),  FLOAT16(55.f),  FLOAT16(56.f),
        FLOAT16(57.f),  FLOAT16(58.f),  FLOAT16(59.f),  FLOAT16(60.f),  FLOAT16(61.f),  FLOAT16(62.f),  FLOAT16(63.f),  FLOAT16(64.f),
        FLOAT16(65.f),  FLOAT16(66.f),  FLOAT16(67.f),  FLOAT16(68.f),  FLOAT16(69.f),  FLOAT16(70.f),  FLOAT16(71.f),  FLOAT16(72.f),

        FLOAT16(73.f),  FLOAT16(74.f),  FLOAT16(75.f),  FLOAT16(76.f),  FLOAT16(77.f),  FLOAT16(78.f),  FLOAT16(79.f),  FLOAT16(80.f),
        FLOAT16(81.f),  FLOAT16(82.f),  FLOAT16(83.f),  FLOAT16(84.f),  FLOAT16(85.f),  FLOAT16(86.f),  FLOAT16(87.f),  FLOAT16(88.f),
        FLOAT16(89.f),  FLOAT16(90.f),  FLOAT16(91.f),  FLOAT16(92.f),  FLOAT16(93.f),  FLOAT16(94.f),  FLOAT16(95.f),  FLOAT16(96.f),


        FLOAT16(97.f),  FLOAT16(98.f),  FLOAT16(99.f),  FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f),
        FLOAT16(105.f), FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f), FLOAT16(112.f),
        FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f), FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),

        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f), FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f),
        FLOAT16(129.f), FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f), FLOAT16(136.f),
        FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f), FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f)
    });

    set_values(input2, {
        0.f, 0.f, 0.f,
        3.f, -3.f, 0.f,

        1.f, -3.f, 1.f,
        -2.f, 0.f, 3.f,

        -1.f, 1.f, 0.f,
        2.f, 0.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 3, 3, 2}, batch_dim, negative_indexes)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f,    2.f,    9.f,    10.f,   17.f,   18.f,   31.f,   32.f,
        35.f,   36.f,   41.f,   42.f,   51.f,   52.f,   59.f,   60.f,
        67.f,   68.f,   77.f,   78.f,   81.f,   82.f,   95.f,   96.f,
        103.f,  104.f,  107.f,  108.f,  113.f,  114.f,  125.f,  126.f,
        129.f,  130.f,  139.f,  140.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


TEST(gather7_gpu_fp16, d222_axisX_bdim_m1) {
    //  Dictionary : 2x2x2x2x2x2
    //  Indexes : 2x2x2x1
    //  Axis : 5
    //  batch_dim : -1
    //  Output : 2x2x2x2x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 1.f 0.f 0.f 0.f 0.f 1.f 0.f
    //
    //  Dictionary:
    //  1.f   2.f   3.f   4.f   5.f   6.f   7.f   8.f   9.f   10.f  11.f  12.f  13.f  14.f  15.f  16.f  17.f  18.f
    //  19.f  20.f  21.f  22.f  23.f  24.f  25.f  26.f  27.f  28.f  29.f  30.f  31.f  32.f  33.f  34.f  35.f  36.f
    //  37.f  38.f  39.f  40.f  41.f  42.f  43.f  44.f  45.f  46.f  47.f  48.f  49.f  50.f  51.f  52.f  53.f  54.f
    //  55.f  56.f  57.f  58.f  59.f  60.f  61.f  62.f  63.f  64.f
    //
    //  Output:
    //  1.f,  3.f,  5.f,  7.f,  10.f, 12.f, 14.f, 16.f,
    //  17.f, 19.f, 21.f, 23.f, 25.f, 27.f, 29.f, 31.f,
    //  33.f, 35.f, 37.f, 39.f, 41.f, 43.f, 45.f, 47.f,
    //  50.f, 52.f, 54.f, 56.f, 57.f, 59.f, 61.f, 63.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, tensor{ 2, 2, 2, 2, 2, 2} }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 2 } }); // Indexes
    int64_t axis = 5;
    int64_t batch_dim = -1;

    set_values(input1, {
        FLOAT16(1.f),   FLOAT16(2.f),   FLOAT16(3.f),   FLOAT16(4.f),   FLOAT16(5.f),   FLOAT16(6.f),   FLOAT16(7.f),   FLOAT16(8.f),
        FLOAT16(9.f),   FLOAT16(10.f),  FLOAT16(11.f),  FLOAT16(12.f),  FLOAT16(13.f),  FLOAT16(14.f),  FLOAT16(15.f),  FLOAT16(16.f),

        FLOAT16(17.f),  FLOAT16(18.f),  FLOAT16(19.f),  FLOAT16(20.f),  FLOAT16(21.f),  FLOAT16(22.f),  FLOAT16(23.f),  FLOAT16(24.f),
        FLOAT16(25.f),  FLOAT16(26.f),  FLOAT16(27.f),  FLOAT16(28.f),  FLOAT16(29.f),  FLOAT16(30.f),  FLOAT16(31.f),  FLOAT16(32.f),

        FLOAT16(33.f),  FLOAT16(34.f),  FLOAT16(35.f),  FLOAT16(36.f),  FLOAT16(37.f),  FLOAT16(38.f),  FLOAT16(39.f),  FLOAT16(40.f),
        FLOAT16(41.f),  FLOAT16(42.f),  FLOAT16(43.f),  FLOAT16(44.f),  FLOAT16(45.f),  FLOAT16(46.f),  FLOAT16(47.f),  FLOAT16(48.f),

        FLOAT16(49.f),  FLOAT16(50.f),  FLOAT16(51.f),  FLOAT16(52.f),  FLOAT16(53.f),  FLOAT16(54.f),  FLOAT16(55.f),  FLOAT16(56.f),
        FLOAT16(57.f),  FLOAT16(58.f),  FLOAT16(59.f),  FLOAT16(60.f),  FLOAT16(61.f),  FLOAT16(62.f),  FLOAT16(63.f),  FLOAT16(64.f),
    });

    set_values(input2, {
        0.f, 1.f,
        0.f, 0.f,

        0.f, 0.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2, 2, 2}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f,  3.f,  5.f,  7.f,  10.f, 12.f, 14.f, 16.f,
        17.f, 19.f, 21.f, 23.f, 25.f, 27.f, 29.f, 31.f,
        33.f, 35.f, 37.f, 39.f, 41.f, 43.f, 45.f, 47.f,
        50.f, 52.f, 54.f, 56.f, 57.f, 59.f, 61.f, 63.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d323_axisY_bdim_m1) {
    //  Dictionary : 3x2x3x4x2
    //  Indexes : 3x2x3x1
    //  Axis : 3
    //  batch_dim : -1
    //  Output : 3x2x3x1x2
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 0.f 3.f 1.f 0.f 1.f 1.f 1.f 2.f 0.f 3.f 3.f 1.f 0.f 2.f 0.f 1.f
    //
    //  Dictionary:
    //  1.f   2.f   3.f   4.f   5.f   6.f   7.f   8.f   9.f   10.f  11.f  12.f  13.f  14.f  15.f  16.f  17.f  18.f
    //  19.f  20.f  21.f  22.f  23.f  24.f  25.f  26.f  27.f  28.f  29.f  30.f  31.f  32.f  33.f  34.f  35.f  36.f
    //  37.f  38.f  39.f  40.f  41.f  42.f  43.f  44.f  45.f  46.f  47.f  48.f  49.f  50.f  51.f  52.f  53.f  54.f
    //  55.f  56.f  57.f  58.f  59.f  60.f  61.f  62.f  63.f  64.f  65.f  66.f  67.f  68.f  69.f  70.f  71.f  72.f
    //  73.f  74.f  75.f  76.f  77.f  78.f  79.f  80.f  81.f  82.f  83.f  84.f  85.f  86.f  87.f  88.f  89.f  90.f
    //  91.f  92.f  93.f  94.f  95.f  96.f  97.f  98.f  99.f  100.f 101.f 102.f 103.f 104.f 105.f 106.f 107.f 108.f
    //  109.f 110.f 111.f 112.f 113.f 114.f 115.f 116.f 117.f 118.f 119.f 120.f 121.f 122.f 123.f 124.f 125.f 126.f
    //  127.f 128.f 129.f 130.f 131.f 132.f 133.f 134.f 135.f 136.f 137.f 138.f 139.f 140.f 141.f 142.f 143.f 144.f
    //
    //  Output:
    //  1.f,   2.f,   9.f,   10.f,  17.f,  18.f,  31.f,  32.f,
    //  35.f,  36.f,  41.f,  42.f,  51.f,  52.f,  59.f,  60.f,
    //  67.f,  68.f,  77.f,  78.f,  81.f,  82.f,  95.f,  96.f,
    //  103.f, 104.f, 107.f, 108.f, 113.f, 114.f, 125.f, 126.f,
    //  129.f, 130.f, 139.f, 140.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 3, 2, 2, 4, 3} }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 3 } }); // Indexes
    int64_t axis = 3;
    int64_t batch_dim = -1;

    set_values(input1, {
        FLOAT16(1.f),   FLOAT16(2.f),   FLOAT16(3.f),   FLOAT16(4.f),   FLOAT16(5.f),   FLOAT16(6.f),   FLOAT16(7.f),   FLOAT16(8.f),
        FLOAT16(9.f),   FLOAT16(10.f),  FLOAT16(11.f),  FLOAT16(12.f),  FLOAT16(13.f),  FLOAT16(14.f),  FLOAT16(15.f),  FLOAT16(16.f),
        FLOAT16(17.f),  FLOAT16(18.f),  FLOAT16(19.f),  FLOAT16(20.f),  FLOAT16(21.f),  FLOAT16(22.f),  FLOAT16(23.f),  FLOAT16(24.f),

        FLOAT16(25.f),  FLOAT16(26.f),  FLOAT16(27.f),  FLOAT16(28.f),  FLOAT16(29.f),  FLOAT16(30.f),  FLOAT16(31.f),  FLOAT16(32.f),
        FLOAT16(33.f),  FLOAT16(34.f),  FLOAT16(35.f),  FLOAT16(36.f),  FLOAT16(37.f),  FLOAT16(38.f),  FLOAT16(39.f),  FLOAT16(40.f),
        FLOAT16(41.f),  FLOAT16(42.f),  FLOAT16(43.f),  FLOAT16(44.f),  FLOAT16(45.f),  FLOAT16(46.f),  FLOAT16(47.f),  FLOAT16(48.f),


        FLOAT16(49.f),  FLOAT16(50.f),  FLOAT16(51.f),  FLOAT16(52.f),  FLOAT16(53.f),  FLOAT16(54.f),  FLOAT16(55.f),  FLOAT16(56.f),
        FLOAT16(57.f),  FLOAT16(58.f),  FLOAT16(59.f),  FLOAT16(60.f),  FLOAT16(61.f),  FLOAT16(62.f),  FLOAT16(63.f),  FLOAT16(64.f),
        FLOAT16(65.f),  FLOAT16(66.f),  FLOAT16(67.f),  FLOAT16(68.f),  FLOAT16(69.f),  FLOAT16(70.f),  FLOAT16(71.f),  FLOAT16(72.f),

        FLOAT16(73.f),  FLOAT16(74.f),  FLOAT16(75.f),  FLOAT16(76.f),  FLOAT16(77.f),  FLOAT16(78.f),  FLOAT16(79.f),  FLOAT16(80.f),
        FLOAT16(81.f),  FLOAT16(82.f),  FLOAT16(83.f),  FLOAT16(84.f),  FLOAT16(85.f),  FLOAT16(86.f),  FLOAT16(87.f),  FLOAT16(88.f),
        FLOAT16(89.f),  FLOAT16(90.f),  FLOAT16(91.f),  FLOAT16(92.f),  FLOAT16(93.f),  FLOAT16(94.f),  FLOAT16(95.f),  FLOAT16(96.f),


        FLOAT16(97.f),  FLOAT16(98.f),  FLOAT16(99.f),  FLOAT16(100.f), FLOAT16(101.f), FLOAT16(102.f), FLOAT16(103.f), FLOAT16(104.f),
        FLOAT16(105.f), FLOAT16(106.f), FLOAT16(107.f), FLOAT16(108.f), FLOAT16(109.f), FLOAT16(110.f), FLOAT16(111.f), FLOAT16(112.f),
        FLOAT16(113.f), FLOAT16(114.f), FLOAT16(115.f), FLOAT16(116.f), FLOAT16(117.f), FLOAT16(118.f), FLOAT16(119.f), FLOAT16(120.f),

        FLOAT16(121.f), FLOAT16(122.f), FLOAT16(123.f), FLOAT16(124.f), FLOAT16(125.f), FLOAT16(126.f), FLOAT16(127.f), FLOAT16(128.f),
        FLOAT16(129.f), FLOAT16(130.f), FLOAT16(131.f), FLOAT16(132.f), FLOAT16(133.f), FLOAT16(134.f), FLOAT16(135.f), FLOAT16(136.f),
        FLOAT16(137.f), FLOAT16(138.f), FLOAT16(139.f), FLOAT16(140.f), FLOAT16(141.f), FLOAT16(142.f), FLOAT16(143.f), FLOAT16(144.f)
    });

    set_values(input2, {
        0.f, 0.f, 0.f,
        3.f, 1.f, 0.f,

        1.f, 1.f, 1.f,
        2.f, 0.f, 3.f,

        3.f, 1.f, 0.f,
        2.f, 0.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 3, 3, 2}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f,   2.f,   9.f,   10.f,  17.f,  18.f,  31.f,  32.f,
        35.f,  36.f,  41.f,  42.f,  51.f,  52.f,  59.f,  60.f,
        67.f,  68.f,  77.f,  78.f,  81.f,  82.f,  95.f,  96.f,
        103.f, 104.f, 107.f, 108.f, 113.f, 114.f, 125.f, 126.f,
        129.f, 130.f, 139.f, 140.f };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d44_axisY_bdim1) {
    //  Dictionary : 4x3x5x1
    //  Indexes : 4x4x1x1
    //  Axis : 2
    //  batch_dim : 1
    //  Output : 4x3x4x1
    //  Input values in fp16

    //  Indexes:
    //  3.f 2.f 3.f 4.f 3.f 2.f 2.f 1.f 1.f 1.f 0.f 4.f 2.f 4.f 3.f 2.f
    //
    //  Dictionary:
    //  84.f  7.f 10.f 69.f 13.f 47.f 75.f  8.f 65.f 28.f  5.f 12.f 56.f 54.f  9.f 31.f 12.f 71.f
    //  55.f  8.f 73.f 16.f 29.f 81.f 81.f 75.f  8.f 74.f 75.f 51.f  7.f 29.f  6.f 72.f 18.f 38.f
    //  54.f 19.f 70.f 16.f 74.f 40.f 72.f 88.f 24.f 14.f 75.f 74.f 82.f 25.f 48.f 13.f 71.f 92.f
    //  9.f 73.f  8.f 80.f 27.f 64.f
    //
    //  Output:
    //  69.f 10.f 69.f 13.f 65.f  8.f 65.f 28.f 54.f 56.f 54.f  9.f 55.f 71.f 71.f 12.f 81.f 29.f
    //  29.f 16.f 75.f 74.f 74.f  8.f 29.f 29.f  7.f 18.f 54.f 54.f 38.f 16.f 40.f 40.f 74.f 24.f
    //  74.f 25.f 82.f 74.f 71.f  9.f 92.f 71.f 80.f 64.f 27.f 80.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 4, 3, 1, 5 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 4, 4, 1, 1 } }); // Indexes
    int64_t axis = 2;
    int64_t batch_dim = 1;

    set_values(input1, {
        FLOAT16(84.f), FLOAT16( 7.f), FLOAT16(10.f), FLOAT16(69.f), FLOAT16(13.f),
        FLOAT16(47.f), FLOAT16(75.f), FLOAT16( 8.f), FLOAT16(65.f), FLOAT16(28.f),
        FLOAT16( 5.f), FLOAT16(12.f), FLOAT16(56.f), FLOAT16(54.f), FLOAT16( 9.f),

        FLOAT16(31.f), FLOAT16(12.f), FLOAT16(71.f), FLOAT16(55.f), FLOAT16( 8.f),
        FLOAT16(73.f), FLOAT16(16.f), FLOAT16(29.f), FLOAT16(81.f), FLOAT16(81.f),
        FLOAT16(75.f), FLOAT16( 8.f), FLOAT16(74.f), FLOAT16(75.f), FLOAT16(51.f),

        FLOAT16( 7.f), FLOAT16(29.f), FLOAT16( 6.f), FLOAT16(72.f), FLOAT16(18.f),
        FLOAT16(38.f), FLOAT16(54.f), FLOAT16(19.f), FLOAT16(70.f), FLOAT16(16.f),
        FLOAT16(74.f), FLOAT16(40.f), FLOAT16(72.f), FLOAT16(88.f), FLOAT16(24.f),

        FLOAT16(14.f), FLOAT16(75.f), FLOAT16(74.f), FLOAT16(82.f), FLOAT16(25.f),
        FLOAT16(48.f), FLOAT16(13.f), FLOAT16(71.f), FLOAT16(92.f), FLOAT16( 9.f),
        FLOAT16(73.f), FLOAT16( 8.f), FLOAT16(80.f), FLOAT16(27.f), FLOAT16(64.f)
    });

    set_values(input2, {
        3.f, 2.f, 3.f, 4.f,
        3.f, 2.f, 2.f, 1.f,
        1.f, 1.f, 0.f, 4.f,
        2.f, 4.f, 3.f, 2.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{4, 3, 4, 1}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        69.f, 10.f, 69.f, 13.f,
        65.f,  8.f, 65.f, 28.f,
        54.f, 56.f, 54.f,  9.f,

        55.f, 71.f, 71.f, 12.f,
        81.f, 29.f, 29.f, 16.f,
        75.f, 74.f, 74.f,  8.f,

        29.f, 29.f,  7.f, 18.f,
        54.f, 54.f, 38.f, 16.f,
        40.f, 40.f, 74.f, 24.f,

        74.f, 25.f, 82.f, 74.f,
        71.f,  9.f, 92.f, 71.f,
        80.f, 64.f, 27.f, 80.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim_m1) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : -1
    //  Output : 3x2x1x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;
    size_t batch_dim = 1;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 1, 1}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 1.f,
        4.f, 3.f,
        5.f, 5.f,
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim1) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : 1
    //  Output : 3x2x1x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;
    int64_t batch_dim = 1;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 1, 1}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 1.f, 4.f,
        3.f, 5.f, 5.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather7_gpu_fp16, d32_axisF_bdim0) {
    //  Dictionary : 3x2x1x1
    //  Indexes : 3x2x1x1
    //  Axis : 1
    //  batch_dim : 0
    //  Output : 3x3x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f 0.f 1.f 0.f 0.f 0.f
    //
    //  Dictionary:
    //  1.f 2.f 3.f 4.f 5.f 6.f
    //
    //  Output:
    //  1.f 1.f 4.f 3.f 5.f 5.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;
    size_t batch_dim = 0;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f),
        FLOAT16(3.f), FLOAT16(4.f),
        FLOAT16(5.f), FLOAT16(6.f)
    });

    set_values(input2, {
        0.f, 0.f, 1.f,
        0.f, 0.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 3, 2, 1}, batch_dim)
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 1.f,
        2.f, 1.f,
        1.f, 1.f,

        3.f, 3.f,
        4.f, 3.f,
        3.f, 3.f,

        5.f, 5.f,
        6.f, 5.f,
        5.f, 5.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
        FLOAT16(1.0f), FLOAT16(2.0f),
        FLOAT16(3.0f), FLOAT16(4.0f)
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{1, 4, 2, 1})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 3, 2, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f,
        2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 2, 1, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 2;

    set_values(input1, {
        FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
        FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

        FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
        FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp16, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp16

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 2, 3, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;

    set_values(input1, {
            FLOAT16(1.f), FLOAT16(2.f), FLOAT16(3.f),
            FLOAT16(4.f), FLOAT16(5.f), FLOAT16(6.f),

            FLOAT16(7.f), FLOAT16(8.f), FLOAT16(9.f),
            FLOAT16(10.f), FLOAT16(11.f), FLOAT16(12.f)
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}

TEST(gather_gpu_fp32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 1.f, 0.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 1, 4, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    set_values(input2, {
        0.f, 1.f,
        1.f, 0.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{1, 4, 2, 1})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 2;

    set_values(input1, {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,

        7.f, 8.f, 9.f,
        10.f, 11.f, 12.f
    });

    set_values(input2, {
        0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in fp32

    //  Indexes:
    //  0.f, 1.f, 2.f, 1.f
    //
    //  Dictionary:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
    //  7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    //
    //  Output:
    //  1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 3, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
    });

    set_values(input2, {
            0.f, 1.f, 2.f, 1.f
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d22_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 3, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 9, 10
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d14_axisB) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 1x4x1x1
    //  Axis : 0
    //  Output : 1x4x2x1
    //  Input values in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 2, 3, 4, 3, 4, 1, 2

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 1, 4, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
            1, 2,
            3, 4
    });

    set_values(input2, {
            0, 1,
            1, 0
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{1, 4, 2, 1})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
            1, 2, 3, 4, 3, 4, 1, 2
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d222_axisB) {
    //  Dictionary : 3x2x2x1
    //  Indexes : 2x2x1x1
    //  Axis : 0
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 3, 2, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 2;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));

    auto p = ov::PartialShape{2, 2, 2, 2};
    std::cout << "[PartialShape]" << p << std::endl;
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 2, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
            1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_int32, dynamic_d22_axisY) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 2x2x1x1
    //  Axis : 2
    //  Output : 2x2x2x2
    //  Input values in i32

    //  Indexes:
    //  0, 1, 2, 1
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 2;

    set_values(input1, {
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12
    });

    set_values(input2, {
            0, 1, 2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", layout{data_types::i32, format::bfyx, ov::PartialShape::dynamic()}));
    topology.add(
            gather("gather", "InputDictionary", "InputText", axis, format::bfwzyx, ov::PartialShape::dynamic())
    );

    network network(engine, topology);
    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
            1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]);
    }
}

TEST(gather_gpu_fp32, d41_axisB) {
    //  Dictionary : 2x2x3x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 4x1x2x3
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 1, 1, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //
    //  Output:
    //  1, 2, 3, 4, 5, 6,
    //  7, 8, 9, 10, 11, 12
    //  7, 8, 9, 10, 11, 12
    //  1, 2, 3, 4, 5, 6,

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 3 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 4, 1, 1, 1 } }); // Indexes
    int64_t axis = 0;

    set_values(input1, {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,

            7.f, 8.f, 9.f,
            10.f, 11.f, 12.f
               });

    set_values(input2, {
            0, 1, 1, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{4, 1, 2, 3})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, d41_axisF) {
    //  Dictionary : 2x3x2x1
    //  Indexes : 4x1x1x1
    //  Axis : 0
    //  Output : 2x4x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  1, 0, 1, 2
    //
    //  Dictionary:
    //  1, 2,   3, 4,   5, 6,
    //  7, 8,   9, 10,  11, 12
    //
    //  Output:
    //  3, 4,   1, 2,   3, 4,   5, 6,
    //  9, 10,  7, 8,   9, 10,  11, 12

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 3, 1, 2 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 4, 1, 1, 1 } }); // Indexes
    int64_t axis = 1;

    set_values(input1, {
            1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            7.f, 8.f, 9.f, 10.f, 11.f, 12.f
               });

    set_values(input2, {
            1, 0, 1, 2
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 4, 1, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            3.f, 4.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f,
            9.f, 10.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, d2_axisX) {
    //  Dictionary : 2x2x1x1
    //  Indexes : 2x1x1x1
    //  Axis : 0
    //  Output : 2x2x1x2
    //  Input values in fp32, indices in i32

    //  Indexes:
    //  0, 0
    //
    //  Dictionary:
    //  1, 2, 3, 4
    //
    //  Output:
    //  1, 1, 2, 2, 3, 3, 4, 4

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 1, 1, 1 } }); // Indexes
    int64_t axis = 3;

    set_values(input1, {
            1.f, 2.f,
            3.f, 4.f,
               });

    set_values(input2, {
            0, 0
               });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{2, 2, 1, 2})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
            1.f, 1.f, 2.f, 2.f,
            3.f, 3.f, 4.f, 4.f
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << " at i=" << i;
    }
}

TEST(gather_gpu_fp32, 322_axisF) {
    //  Dictionary : 3x3x1x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 3x2x2x1
    //  Input values in i32

    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 3, 3, 1, 1 } }); // data
    auto input2 = engine.allocate_memory({ data_types::i32, format::bfyx, tensor{ 2, 2, 1, 1 } }); // Indexes
    int64_t axis = 1;

    set_values(input1, {
        0, 1, 2,  10, 11, 12,   20, 21, 22
    });

    set_values(input2, {
        1, 0,
        2, 1
    });

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 2, 1})
    );

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<int> output_ptr(output, get_test_stream());

    std::vector<int> expected_results = {
        1, 0, 2, 1,   11, 10, 12, 11,   21, 20, 22, 21
    };

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << i;
    }
}

TEST(gather_gpu_u8, 322_axisF) {
    //  Dictionary : 3x3x1x1
    //  Indexes : 2x2x1x1
    //  Axis : 1
    //  Output : 3x2x2x1
    //  Input values in u8

    auto &engine = get_test_engine();

    auto input1 = engine.allocate_memory({data_types::u8, format::bfyx, tensor{3, 3, 1, 1}}); // data
    auto input2 = engine.allocate_memory({data_types::i32, format::bfyx, tensor{2, 2, 1, 1}}); // Indexes
    int64_t axis = 1;

    set_values<uint8_t>(input1, {0, 1, 2, 10, 11, 12, 20, 21, 22});

    set_values(input2, {1, 0,
                        2, 1});

    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather("gather", "InputDictionary", "InputText", axis, ov::Shape{3, 2, 2, 1}));

    network network(engine, topology);

    network.set_input_data("InputDictionary", input1);
    network.set_input_data("InputText", input2);

    auto outputs = network.execute();

    auto output = outputs.at("gather").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    std::vector<uint8_t> expected_results = {
        1, 0, 2, 1, 11, 10, 12, 11, 21, 20, 22, 21};

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], output_ptr[i]) << i;
    }
}
