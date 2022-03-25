// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>

#include "ngraph/runtime/reference/gather_elements.hpp"

using namespace cldnn;
using namespace ::tests;

inline void DoTestBase(engine& engine,
    const cldnn::memory::ptr input0,
    const cldnn::memory::ptr input1,
    const std::vector<float>& expected_results,
    const int axis,
    const cldnn::format fmt,
    const tensor ts,
    const bool batch_merged_output = true) {
    topology topology;

    int input_rank = 0;
    if (input0->get_layout().format == format::bfyx) {
        input_rank = 4;
    } else if (input0->get_layout().format == format::bfzyx) {
        input_rank = 5;
    } else if (input0->get_layout().format == format::bfwzyx) {
        input_rank = 6;
    } else {
        FAIL();
    }

    auto gather_elements_inst = gather_elements("gather_elements", "InputData", "InputIndices", axis);
    topology.add(input_layout("InputData", input0->get_layout()));
    topology.add(input_layout("InputIndices", input1->get_layout()));
    topology.add(gather_elements_inst);

    network network(engine, topology);

    network.set_input_data("InputData", input0);
    network.set_input_data("InputIndices", input1);
    auto outputs = network.execute();
    auto output = outputs.at("gather_elements").get_memory();

    // Compare output shape
    auto output_format = output->get_layout().format;
    auto output_shape = output->get_layout().size;

    EXPECT_EQ(fmt, output_format);

    int32_t dim_size = 6;
    if (fmt == format::bfyx) {
        dim_size = 4;
    } else if (fmt == format::bfzyx) {
        dim_size = 5;
    }

    for (int32_t i = 0; i < dim_size; i++)
    {
        EXPECT_EQ(ts.sizes()[i], output_shape.sizes()[i]);
    }

    // Compare output value
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < expected_results.size(); ++i)
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    // for(int i = 0; i < std::min<size_t>(100, expected_results.size()); ++i){
    //     if (i % output->get_layout().get_dims().back() == 0)
    //         std::cout << std::endl;
    //     std::cout << float16_to_float32(output_ptr[i]) << ' ';
    // }
}

TEST(gather_elements_gpu_fp16, d23425_i2339_a3){
    int axis=2;

    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 5, 2, 4 } }); // data
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfzyx, { 2, 3, 5, 2, 2 } }); // indices

    int axis_len=input0->get_layout().get_dim(axis);

    auto ivec0=generate_random_1d<FLOAT16>(input0->count(),0,999);
    auto ivec1=generate_random_1d<int>(input1->count(),-axis_len,axis_len-1);
    set_values(input0, ivec0);
    set_values(input1, ivec1);

    std::vector<size_t> ivec1u(ivec1.size());
    std::transform(
        ivec1.begin(),
        ivec1.end(),
        ivec1u.begin(),
        [axis_len](int idx){return idx<0?idx+axis_len:idx;}
    );

    std::vector<FLOAT16> expected16(input1->count());
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};
    ngraph::runtime::reference::gather_elements<FLOAT16,size_t>(
        ivec0.data(),
        ivec1u.data(),
        expected16.data(),
        ov::Shape(to_vec_size_t(input0->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        axis);
    std::vector<float> expected32;
    for(auto&i:expected16)
        expected32.push_back(float16_to_float32(i.v));
    
    // for(int i = 0; i < std::min<size_t>(100, expected16.size()); ++i){
    //     if (i % input1->get_layout().get_dims().back() == 0)
    //         std::cout << std::endl;
    //     std::cout << expected32[i] << ' ';
    // }

    DoTestBase(engine, input0, input1, expected32, axis, format::bfzyx, input1->get_layout().size);
}

TEST(gather_elements_gpu_fp16, d2342_i1342_a0){
    int axis=0;

    auto& engine = get_test_engine();

    //NOTE: format::bfyx, {2,3,2,4}로 하면 b=2,f=3,x=2,y=4로 저장된다. 아래 코드 실행해보면 이해됨
    /*
        for(auto i:input0->get_layout().get_ordered_dims())std::cout<<i<<' ';
        std::cout<<std::endl;
        for(auto i:input1->get_layout().get_ordered_dims())std::cout<<i<<' ';
        std::cout<<std::endl;
        std::cout<<input0->get_layout().size<<std::endl;
    */
    //차원을 내부적으로는 b,f,x,y,z,w,g 순서로 표현한다는듯.
    //다만 차원순서와 달리 메모리포맷은 bfyx로 잘 저장된다.
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 2, 4 } }); // data
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 1, 3, 2, 4 } }); // indices
    // auto input2 = engine.allocate_memory({  }); //axis
    // 최적화를 위해서 axis매개변수를 넘기지 말고 그냥 6개 케이스에 대해 다 나눠서 만들기?
    // // expected output dim: v6{1,3,2,4}

    int axis_len=input0->get_layout().get_dim(axis);

    auto ivec0=generate_random_1d<FLOAT16>(input0->count(),0,999);
    auto ivec1=generate_random_1d<int>(input1->count(),-axis_len,axis_len-1);
    set_values(input0, ivec0);
    set_values(input1, ivec1);

    std::vector<size_t> ivec1u(ivec1.size());
    std::transform(
        ivec1.begin(),
        ivec1.end(),
        ivec1u.begin(),
        [axis_len](int idx){return idx<0?idx+axis_len:idx;}
    );

    std::vector<FLOAT16> expected16(input1->count());
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};
    ngraph::runtime::reference::gather_elements<FLOAT16,size_t>(
        ivec0.data(),
        ivec1u.data(),
        expected16.data(),
        ov::Shape(to_vec_size_t(input0->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        axis);
    std::vector<float> expected32;
    for(auto&i:expected16)
        expected32.push_back(float16_to_float32(i.v));

    DoTestBase(engine, input0, input1, expected32, axis, format::bfyx, input1->get_layout().size);
}

TEST(gather_elements_gpu_fp16, d2334_i2339_a3){
    int axis=3;

    auto& engine = get_test_engine();

    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 4, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { 2, 3, 9, 3 } }); // indices

    int axis_len=input0->get_layout().get_dim(axis);

    auto ivec0=generate_random_1d<FLOAT16>(input0->count(),0,999);
    auto ivec1=generate_random_1d<int>(input1->count(),-axis_len,axis_len-1);
    set_values(input0, ivec0);
    set_values(input1, ivec1);

    std::vector<size_t> ivec1u(ivec1.size());
    std::transform(
        ivec1.begin(),
        ivec1.end(),
        ivec1u.begin(),
        [axis_len](int idx){return idx<0?idx+axis_len:idx;}
    );

    std::vector<FLOAT16> expected16(input1->count());
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};
    ngraph::runtime::reference::gather_elements<FLOAT16,size_t>(
        ivec0.data(),
        ivec1u.data(),
        expected16.data(),
        ov::Shape(to_vec_size_t(input0->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),
        axis);
    std::vector<float> expected32;
    for(auto&i:expected16)
        expected32.push_back(float16_to_float32(i.v));

    DoTestBase(engine, input0, input1, expected32, axis, format::bfyx, input1->get_layout().size);
}

//A: indicis의 axis=0축 크기
#define TOKENPASTE_(x, y) x ## y
#define TOKENPASTE(x, y) TOKENPASTE_(x, y)
#define TESTbfyx0(B,F,Y,X,A) TEST(gather_elements_gpu_fp16, TOKENPASTE(bfyx0_,__LINE__)){\
    int axis=0;\
    auto& engine = get_test_engine();\
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { B,F,X,Y } });\
    auto input1 = engine.allocate_memory({ data_types::i32, format::bfyx, { A,F,X,Y } });\
\
    auto ivec0=generate_random_1d<FLOAT16>(input0->count(),0,999);\
    auto ivec1=generate_random_1d<int>(input1->count(),-B,B-1);\
    set_values(input0, ivec0);\
    set_values(input1, ivec1);\
\
    std::vector<size_t> ivec1u(ivec1.size());\
    std::transform(ivec1.begin(),ivec1.end(),ivec1u.begin(),[](int idx){return idx<0?idx+B:idx;});\
\
    std::vector<FLOAT16> expected16(input1->count());\
    auto to_vec_size_t=[](const std::vector<int>& vec){return std::vector<size_t>(vec.begin(),vec.end());};\
    ngraph::runtime::reference::gather_elements<FLOAT16,size_t>(\
        ivec0.data(),\
        ivec1u.data(),\
        expected16.data(),\
        ov::Shape(to_vec_size_t(input0->get_layout().get_dims())),\
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),\
        ov::Shape(to_vec_size_t(input1->get_layout().get_dims())),\
        axis);\
    std::vector<float> expected32;\
    for(auto&i:expected16)\
        expected32.push_back(float16_to_float32(i.v));\
\
    DoTestBase(engine, input0, input1, expected32, axis, format::bfyx, input1->get_layout().size);\
}

//extreme cases
// TESTbfyx0(0,0,0,0,0);//zero-able?
TESTbfyx0(1,1,1,1,1);//smallest
TESTbfyx0(1,1,1,1,11111111);//large axis_len
TESTbfyx0(66,66,66,66,1);//huge data
TESTbfyx0(44,44,44,44,44);//balanced big1
TESTbfyx0(55,55,55,55,55);//balanced big2
