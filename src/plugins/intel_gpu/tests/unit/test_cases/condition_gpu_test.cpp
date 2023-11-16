// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "random_generator.hpp"
#include "test_utils.h"
#include "condition_inst.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/condition.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

namespace {
template <class T>
bool is_output_equal(const cldnn::memory::ptr mem, const std::vector<T>& ref) {
    if (mem->count() != ref.size())
        return false;
    cldnn::mem_lock<T> ptr(mem, get_test_stream());
    for (size_t i = 0; i < mem->get_layout().count(); i++) {
        if (!are_equal(ptr[i], ref[i])) return false;
    }
    return true;
}


topology generate_simple_branch (bool branch_true_false, const primitive_id& id, const primitive_id& input_id, const data_types dt = data_types::f32)
{
    topology branch;
    if (branch_true_false) {
        branch.add(
            input_layout(input_id, { dt, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(id + "_when_true", input_id, cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 })
        );
    } else {
        branch.add(
            input_layout(input_id, { dt, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(id + "_when_false", input_id, cldnn::pooling_mode::average, { 1, 2 }, { 1, 2 })
        );
    }
    return branch;
}
}  // namespace

template < typename DataType>
struct condition_data_types {
    using type = DataType;
};

template <typename ConditionDataType>
class condition_gpu_basic_test : public ::testing::Test {
public:

    using input_type = typename ConditionDataType::type;
    std::vector<input_type> convert_data(std::vector<int> in_vec) {
        const size_t vec_size = in_vec.size();
        std::vector<input_type> converted_data_vec(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
            converted_data_vec[i] = (input_type)in_vec[i];
        }
        return converted_data_vec;
    }

    void run_test() {
        auto& engine = get_test_engine();

        auto dat_dt = static_cast<ov::element::Type>(ov::element::from<typename ConditionDataType::type>());

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ dat_dt, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });
        auto scale_mem = engine.allocate_memory({ dat_dt, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id            = "predicate";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";
        primitive_id scale_data_id      = "scale_data";
        primitive_id output_id          = "output";

        condition::branch branch_true;
        {
            cldnn::topology branch_true_topology   = generate_simple_branch(true,  cond_id, branch_input_id, dat_dt);
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id, branch_input_id});
            branch_true.output_map.insert({0, "condi_when_true"});
        }
        condition::branch branch_false;
        {
            cldnn::topology branch_false_topology  = generate_simple_branch(false, cond_id, branch_input_id, dat_dt);
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id, branch_input_id});
            branch_false.output_map.insert({0, "condi_when_false"});
        }

        cldnn::topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            input_layout(scale_data_id, scale_mem->get_layout())
        );
        topology.add(
            condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
        );
        topology.add(
            eltwise(output_id, { input_info(cond_id), input_info(scale_data_id) }, eltwise_mode::prod)
        );

        network net(engine, topology, config);
        set_values(input, convert_data({ 1, 2, 3, 4 }));
        set_values(scale_mem, convert_data({ 10 }));
        net.set_input_data(input_id, input);
        net.set_input_data(scale_data_id, scale_mem);

        decltype(net.execute()) out;

        //WHEN TRUE
        set_values(predicate, { 1 });
        net.set_input_data(pred_id, predicate);
        out = net.execute();
        auto out_data_true = out.at(output_id).get_memory();
        ASSERT_TRUE(is_output_equal(out_data_true, convert_data({ 20, 40 })));

        //WHEN FALSE
        set_values(predicate, { 0 });
        net.set_input_data(pred_id, predicate);
        out = net.execute();
        auto out_data_false = out.at(output_id).get_memory();
        ASSERT_TRUE(is_output_equal(out_data_false, convert_data({ 15, 35 })));
    }
};

using test_data_types = testing::Types<condition_data_types<ov::float16>,
                                    condition_data_types<float>>;

TYPED_TEST_SUITE(condition_gpu_basic_test, test_data_types);

TYPED_TEST(condition_gpu_basic_test, simple_basic_test) {
    this->run_test();
}

TEST(condition_gpu, basic_range_equal_comp) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

    auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    primitive_id condi_id = "condi";
    primitive_id branch_input_id = "branch_input";
    primitive_id concat_id = "concat";

    cldnn::topology topology;
    topology.add(
        input_layout("input0", input0->get_layout())
    );
    topology.add(
        input_layout("input1", input1->get_layout())
    );
    topology.add(
        input_layout("predicate", predicate->get_layout())
    );
    topology.add(
        concatenation("concat", { input_info("input0"), input_info("input1") }, 3)
    );

    condition::branch branch_true;
    {
        cldnn::topology branch_true_topology  = generate_simple_branch(true,  condi_id, branch_input_id);
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({concat_id, branch_input_id});
        branch_true.output_map.insert({0, "condi_when_true"});
    }
    condition::branch branch_false;
    {
        cldnn::topology branch_false_topology = generate_simple_branch(false, condi_id, branch_input_id);
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({concat_id, branch_input_id});
        branch_false.output_map.insert({0, "condi_when_false"});
    }

    topology.add(
        condition("condi", {input_info("predicate"), input_info("concat")}, branch_true, branch_false)
    );

    std::vector<float> input0_data = {
        1, 2
    };
    std::vector<float> input1_data = {
        3, 4
    };
    std::vector<uint8_t> predicate_data_true = {
        1
    };
    std::vector<float> pooling_when_true_data = {
        2, 4
    };
    std::vector<uint8_t> predicate_data_false = {
        0
    };
    std::vector<float> pooling_when_false_data = {
        1.5, 3.5
    };

    set_values(input0, input0_data);
    set_values(input1, input1_data);
    network net(engine, topology, config);
    net.set_input_data("input0", input0);
    net.set_input_data("input1", input1);

    decltype(net.execute()) outputs;

    //CHECK TRUE
    set_values(predicate, predicate_data_true);
    net.set_input_data("predicate", predicate);
    outputs = net.execute();

    auto out_data_true = outputs.at("condi").get_memory();
    ASSERT_TRUE(is_output_equal(out_data_true, pooling_when_true_data));

    //CHECK FALSE
    set_values(predicate, predicate_data_false);
    net.set_input_data("predicate", predicate);
    outputs = net.execute();

    auto out_data_false = outputs.at("condi").get_memory();
    ASSERT_TRUE(is_output_equal(out_data_false, pooling_when_false_data));
}

TEST(condition_gpu, dynamic_shapes) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    const int64_t d1 = 2;
    const int64_t d2 = 4;
    layout input_lay = {{-1, d1, -1, d2}, data_types::f32, format::bfyx};

    auto predicate = engine.allocate_memory({{ 1 }, data_types::u8, format::bfyx });

    const primitive_id condition_id = "condition";
    const primitive_id condition_id_true = condition_id + "_when_true";
    const primitive_id condition_id_false = condition_id + "_when_false";
    const primitive_id branch_input_id = "branch_input";
    const primitive_id model_input = "input";
    const primitive_id predicate_input = "predicate";
    const primitive_id tranpose = "transpose";

    cldnn::topology topology;
    topology.add(input_layout(model_input, input_lay));
    topology.add(input_layout(predicate_input, predicate->get_layout()));
    topology.add(permute(tranpose, model_input, {1, 0, 2, 3}));
    const float shift = 4.f;

    auto generate_simple_branch = [&](bool branch_true_false, const primitive_id& input_id, const data_types dt) {
        auto mem = engine.allocate_memory(layout{{d1, 1, 1, d2}, dt, format::bfyx});
        {
            cldnn::mem_lock<float> l(mem, get_test_stream());
            for (size_t i = 0; i < mem->count(); i++) {
                l.data()[i] = shift;
            }
        }

        primitive_id const_id = "const_input";
        eltwise_mode mode = branch_true_false ? eltwise_mode::sum : eltwise_mode::sub;
        auto id = branch_true_false ? condition_id_true : condition_id_false;
        cldnn::topology branch_topology(input_layout(input_id, { {d1, -1, -1, d2}, dt, format::bfyx }),
                                        data(const_id, mem),
                                        eltwise(id, {input_id, const_id}, mode)
        );
        condition::branch branch;
        branch.inner_program = program::build_program(engine, branch_topology, config, false, false, true);
        branch.input_map.insert({tranpose, branch_input_id});
        branch.output_map.insert({0, id});

        return branch;
    };

    condition::branch branch_true = generate_simple_branch(true, branch_input_id, data_types::f32);
    condition::branch branch_false = generate_simple_branch(false, branch_input_id, data_types::f32);

    topology.add(condition(condition_id, { input_info(predicate_input), tranpose }, branch_true, branch_false));

    tests::random_generator rg(GET_SUITE_NAME);
    std::vector<uint8_t> predicate_data_true = { 1 };
    std::vector<uint8_t> predicate_data_false = { 0 };

    network net(engine, topology, config);

    auto check_output = [](const cldnn::memory::ptr mem, const std::vector<float>& ref, ov::Shape expected_shape) {
        ASSERT_EQ(mem->get_layout().get_shape(), expected_shape);
        ASSERT_EQ(mem->count(), ref.size());
        cldnn::mem_lock<float> ptr(mem, get_test_stream());
        for (size_t i = 0; i < mem->get_layout().count(); i++) {
            ASSERT_EQ(ptr[i], ref[i]) << "i = " << i;
        }
    };

    for (size_t i = 0; i < 10; i++) {
        layout l = {{1, d1, 1 + static_cast<int64_t>(i), d2}, data_types::f32, format::bfyx};
        std::vector<float> input_data = rg.generate_random_1d<float>(l.count(), -10, 10);
        auto mem = engine.allocate_memory(l);
        std::vector<float> expected_result_when_true = input_data;
        std::vector<float> expected_result_when_false = input_data;

        for (size_t i = 0; i < input_data.size(); i++) {
            expected_result_when_true[i] += shift;
            expected_result_when_false[i] -= shift;
        }

        set_values(mem, input_data);
        set_values(predicate, predicate_data_true);
        net.set_input_data(model_input, mem);
        net.set_input_data(predicate_input, predicate);
        auto outputs = net.execute();
        check_output(outputs.at(condition_id).get_memory(), expected_result_when_true, {d1, 1, 1+i, d2});

        set_values(predicate, predicate_data_false);
        net.set_input_data(model_input, mem);
        net.set_input_data(predicate_input, predicate);
        outputs = net.execute();
        check_output(outputs.at(condition_id).get_memory(), expected_result_when_false, {d1, 1, 1+i, d2});
    }
}

TEST(condition_gpu, basic_stacked_ifs) {
    /*
        <prims...>
        <if>
        <...>
        <end_if>
        <...>
        <if>
        <...>
        <end_if>
        <prims...>
    */
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto predicate2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    primitive_id input_id           = "input";
    primitive_id pred_id            = "predicate";
    primitive_id predicate2_id      = "predicate2";
    primitive_id branch_input_id    = "branch_input";
    primitive_id cond_id            = "condi";
    primitive_id cond2_id           = "condi2";
    primitive_id scale_data_id      = "scale_data";
    primitive_id output_id          = "output";

    topology condi_1_true = generate_simple_branch(true, cond_id, branch_input_id);
    topology condi_1_false = generate_simple_branch(false, cond_id, branch_input_id);
    topology condi_2_true;
    condi_2_true.add(
        input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
        activation("activ_when_true", input_info(branch_input_id), activation_func::log2)
    );
    topology condi_2_false;
    condi_2_false.add(
        input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
        activation("activ_when_false", input_info(branch_input_id), activation_func::relu)
    );

    condition::branch branch_condi_1_true;
    branch_condi_1_true.inner_program = program::build_program(engine, condi_1_true, config, false, false, true);
    branch_condi_1_true.input_map.insert({input_id, branch_input_id});
    branch_condi_1_true.output_map.insert({0, "condi_when_true"});

    condition::branch branch_condi_1_false;
    branch_condi_1_false.inner_program = program::build_program(engine, condi_1_false, config, false, false, true);
    branch_condi_1_false.input_map.insert({input_id, branch_input_id});
    branch_condi_1_false.output_map.insert({0, "condi_when_false"});

    condition::branch branch_condi_2_true;
    branch_condi_2_true.inner_program = program::build_program(engine, condi_2_true, config, false, false, true);
    branch_condi_2_true.input_map.insert({cond_id, branch_input_id});
    branch_condi_2_true.output_map.insert({0, "activ_when_true"});

    condition::branch branch_condi_2_false;
    branch_condi_2_false.inner_program = program::build_program(engine, condi_2_false, config, false, false, true);
    branch_condi_2_false.input_map.insert({cond_id, branch_input_id});
    branch_condi_2_false.output_map.insert({0, "activ_when_false"});

    topology topology;
    topology.add(
        input_layout(input_id, input->get_layout())
    );
    topology.add(
        input_layout(pred_id, predicate->get_layout())
    );
    topology.add(
        condition(cond_id, { input_info(pred_id), input_info(input_id) }, branch_condi_1_true, branch_condi_1_false)
    );
    topology.add(
        input_layout(predicate2_id, predicate2->get_layout())
    );
    topology.add(
        condition(cond2_id, { input_info(predicate2_id), input_info(cond_id) }, branch_condi_2_true, branch_condi_2_false)
    );

    std::vector<float> input_data = {
        1, 2, 3, 4
    };
    std::vector<uint8_t> predicate_data = {
        1
    };
    std::vector<uint8_t> predicate_2_data = {
        0
    };
    set_values(input, input_data);
    set_values(predicate, predicate_data);
    set_values(predicate2, predicate_2_data);

    network net(engine, topology, config);
    net.set_input_data(input_id, input);
    net.set_input_data(pred_id, predicate);
    net.set_input_data(predicate2_id, predicate2);
    auto outputs = net.execute();

    std::vector<float> ref_data = {
        2.0f, 4.0f
    };
    auto out_data = outputs.at(cond2_id).get_memory();
    ASSERT_TRUE(is_output_equal(out_data, ref_data));
}

TEST(condition_gpu, basic_nested_ifs) {
    /*
    <prims...>
    <if 0>
    <...>
    <if 1>
    <...>
    <end_if 1>
    <...>
    <end_if 0>
    <prims...>
    */
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto predicate2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_5_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    set_values(scale_5_mem, { 5.0f });
    auto scale_10_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    set_values(scale_10_mem, { 10.0f });

    condition::branch nested_true;
    {
        cldnn::topology nested_true_topology;
        nested_true_topology.add(
            input_layout("branch_input1", { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
            data("scale_5_data", scale_5_mem),
            eltwise("scale_5", { input_info("branch_input1"), input_info("scale_5_data") }, eltwise_mode::prod)
        );
        nested_true.inner_program = program::build_program(engine, nested_true_topology, config, false, false, true);
        nested_true.input_map.insert({"pooling_when_true", "branch_input1"});
        nested_true.output_map.insert({0, "scale_5"});
    }
    condition::branch nested_false;
    {
        cldnn::topology nested_false_topology;
        nested_false_topology.add(
            input_layout("branch_input2", { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
            data("scale_10_data", scale_10_mem),
            eltwise("scale_10", { input_info("branch_input2"), input_info("scale_10_data") }, eltwise_mode::prod)
        );
        nested_false.inner_program = program::build_program(engine, nested_false_topology, config, false, false, true);
        nested_false.input_map.insert({"pooling_when_true", "branch_input2"});
        nested_false.output_map.insert({0, "scale_10"});
    }

    condition::branch branch_true;
    {
        cldnn::topology branch_true_topology;
        branch_true_topology.add(
            input_layout("branch_input3", { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling("pooling_when_true", input_info("branch_input3"), cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 }),
            input_layout("predicate2", predicate2->get_layout()),
            condition( "condi_nested", {input_info("predicate2"), input_info("pooling_when_true")}, nested_true, nested_false)
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({"input", "branch_input3"});
        branch_true.output_map.insert({0, "condi_nested"});
    }

    condition::branch branch_false;
    {
        cldnn::topology branch_false_topology;
        branch_false_topology.add(
            input_layout("branch_input4", { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling("pooling_when_false", input_info("branch_input4"), cldnn::pooling_mode::average, { 1, 2 }, { 1, 2 })
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({"input", "branch_input4"});
        branch_false.output_map.insert({0, "pooling_when_false"});
    }

    cldnn::topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );

    topology.add(
        input_layout("predicate", predicate->get_layout())
    );

    topology.add(
        condition("condi", {input_info("predicate"), input_info("input")}, branch_true, branch_false)
    );

    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> predicate_data = {
        1.0f
    };
    std::vector<float> predicate_2_data = {
        2.0f, 4.0f
    };
    set_values(input, input_data);
    set_values(predicate, predicate_data);
    set_values(predicate2, predicate_2_data);

    network net(engine, topology, config);
    net.set_input_data("input", input);
    net.set_input_data("predicate", predicate);
    net.set_input_data("predicate2", predicate2);
    auto outputs = net.execute();

    auto out_data = outputs.at("condi").get_memory();
    ASSERT_TRUE(is_output_equal(out_data, std::vector<float>({ 10.0f, 20.0f })));
}

TEST(condition_gpu, negative_predicate_wrong_layout) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 5, 1 } });

    primitive_id input_id           = "input";
    primitive_id pred_id            = "predicate";
    primitive_id branch_input_id    = "branch_input";
    primitive_id cond_id            = "condi";

    condition::branch branch_true;
    {
        cldnn::topology branch_true_topology   = generate_simple_branch(true,  cond_id, branch_input_id, data_types::f32);
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({input_id, branch_input_id});
        branch_true.output_map.insert({0, "condi_when_true"});
    }
    condition::branch branch_false;
    {
        cldnn::topology branch_false_topology  = generate_simple_branch(false, cond_id, branch_input_id, data_types::f32);
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id, branch_input_id});
        branch_false.output_map.insert({0, "condi_when_false"});
    }

    topology topology;
    topology.add(
        input_layout(input_id, input->get_layout())
    );
    topology.add(
        input_layout(pred_id, predicate->get_layout())
    );
    topology.add(
        condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
    );

    EXPECT_ANY_THROW(network net(engine, topology, config););
}

TEST(condition_gpu, negative_not_same_layouts) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    primitive_id input_id           = "input";
    primitive_id pred_id            = "predicate";
    primitive_id branch_input_id    = "branch_input";
    primitive_id cond_id            = "condi";

    condition::branch branch_true;
    {
        primitive_id pool_id = "pooling_when_true";
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(pool_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 })
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({input_id, branch_input_id});
        branch_true.output_map.insert({0, pool_id});
    }

    condition::branch branch_false;
    {
        primitive_id pool_id = "pooling_when_false";
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(pool_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 1, 4 }, { 1, 4 })
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id, branch_input_id});
        branch_false.output_map.insert({0, pool_id});
    }


    topology topology;
    topology.add(
        input_layout(input_id, input->get_layout())
    );
    topology.add(
        input_layout(pred_id, predicate->get_layout())
    );
    topology.add(
        condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
    );

    EXPECT_ANY_THROW(network net(engine, topology, config););
}

TEST(condition_gpu, negative_same_names_within_different_networks) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    primitive_id input_id           = "input";
    primitive_id pred_id         = "predicate";
    primitive_id branch_input_id    = "branch_input";
    primitive_id cond_id            = "condi";
    primitive_id duplicated_id      = "pooling_check_name";

    condition::branch branch_true;
    {
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(duplicated_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({input_id, branch_input_id});
        branch_true.output_map.insert({0, duplicated_id});
    }

    condition::branch branch_false;
    {
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling("pooling_when_false", input_info(branch_input_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id, branch_input_id});
        branch_false.output_map.insert({0, "pooling_when_false"});
    }

    topology topology;
    topology.add(
        input_layout(input_id, input->get_layout())
    );
    topology.add(
        input_layout(pred_id, predicate->get_layout())
    );
    topology.add(
        condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
    );
    topology.add(
        pooling(duplicated_id, input_info(cond_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
    );

    EXPECT_ANY_THROW(network net(engine, topology, config););
}

TEST(condition_gpu, empty_body) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto predicate_mem = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

    primitive_id input_id1           = "input1";
    primitive_id input_id2           = "input2";
    primitive_id pred_id             = "predicate";
    primitive_id branch_input_id1    = "branch_input1";
    primitive_id branch_input_id2    = "branch_input2";
    primitive_id cond_id             = "condi";

    condition::branch branch_true;
    {
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout(branch_input_id1, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            input_layout(branch_input_id2, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
            eltwise("eltwise", { input_info(branch_input_id1), input_info(branch_input_id2) }, eltwise_mode::sum)
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({input_id1, branch_input_id1});
        branch_true.input_map.insert({input_id2, branch_input_id2});
        branch_true.output_map.insert({0, "eltwise"});
    }

    condition::branch branch_false;
    {
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout(branch_input_id2, { data_types::f32, format::bfyx, { 1, 1, 4, 1 } }),
            reorder("result", input_info(branch_input_id2), {data_types::f32, format::bfyx, {1, 1, 4, 1}})
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id2, branch_input_id2});
        branch_false.output_map.insert({0, "result"});
    }

    topology topology;
    topology.add(input_layout(input_id1, input_mem->get_layout()));
    topology.add(input_layout(input_id2, input_mem->get_layout()));
    topology.add(input_layout(pred_id, predicate_mem->get_layout()));
    topology.add(condition(cond_id, {input_info(pred_id), input_info(input_id1), input_info(input_id2)}, branch_true, branch_false)
    );

    network net(engine, topology, config);
    ASSERT_TRUE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());
    ASSERT_FALSE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_true().inner_program->can_be_optimized());
}
