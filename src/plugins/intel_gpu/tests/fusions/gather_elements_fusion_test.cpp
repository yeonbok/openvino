// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gather_elements_test_params {
    data_types data_type;

    format input_format;
    tensor input_shape;

    format indices_format;
    tensor indices_shape;

    format output_format;
    tensor output_shape;

    int axis;

    data_types default_type;
    format default_format;

    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherElementsPrimitiveFusingTest : public ::BaseFusingTest<gather_elements_test_params> {
public:
    void execute(gather_elements_test_params& p) {
        // topology topo;
        // topo.add(input_layout("InputData", layout({ data_types::f16, format::bfzyx, { 2, 3, 5, 2, 4 } })));
        // topo.add(input_layout("InputIndices", layout({ data_types::f16, format::bfzyx, { 2, 3, 5, 2, 2 } })));
        // topo.add(gather_elements("gather_elements", "InputData", "InputIndices", 2));
        // network testnet(engine, topo);

        auto input_prim = get_mem(get_input_layout(p));
        // network network_basic(this->engine, this->topology_non_fused);
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.indices_format, p.indices_shape };
    }

    layout get_output_layout(gather_elements_test_params& p) {
        return layout{ p.data_type, p.output_format, p.output_shape };
    }

    layout get_per_channel_layout(gather_elements_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.output_shape.feature[0], 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ GatherElements cases ------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_GATHER_ND_FP16_4D_1 data_types::f16, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_2 data_types::f16, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, 1, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_3 data_types::f16, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 7, 9, 8 }, 3, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP16_5D_1 data_types::f16, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfzyx, { 5, 6, 7, 8, 5 }, 0, data_types::f16, format::bfzyx
#define CASE_GATHER_ND_FP16_5D_2 data_types::f16, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfzyx, { 5, 6, 7, 8, 5 }, 1, data_types::f16, format::bfzyx
#define CASE_GATHER_ND_FP16_5D_3 data_types::f16, format::bfzyx, { 5, 4, 7, 8, 5 }, format::bfzyx, { 5, 4, 7, 8, 5 }, format::bfzyx, { 5, 4, 7, 8, 5 }, 2, data_types::f16, format::bfzyx
#define CASE_GATHER_ND_FP16_5D_4 data_types::f16, format::bfzyx, { 5, 4, 7, 8, 3 }, format::bfzyx, { 5, 4, 7, 8, 3 }, format::bfzyx, { 5, 4, 7, 8, 3 }, 3, data_types::f16, format::bfzyx
#define CASE_GATHER_ND_FP16_5D_5 data_types::f16, format::bfzyx, { 5, 4, 7, 2, 3 }, format::bfzyx, {  5, 4, 7, 2, 3 }, format::bfzyx, {  5, 4, 7, 2, 3 }, 4, data_types::f16, format::bfzyx
#define CASE_GATHER_ND_FP16_5D_6 data_types::f16, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, {  5, 4, 7, 4, 4 }, format::bfzyx, {  5, 4, 7, 4, 4 }, 2, data_types::f16, format::bfzyx

// #define CASE_GATHER_ND_FP16_6D_1 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 8, 5 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 20, 2, 6, 7 }, 5, 4, 2, data_types::f16, format::bfyx
// #define CASE_GATHER_ND_FP16_6D_2 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 40, 6, 1, 1 }, 5, 4, 3, data_types::f16, format::bfyx
// #define CASE_GATHER_ND_FP16_6D_3 data_types::f16, format::bfwzyx, { 5, 4, 6, 7, 2, 2 }, format::bfzyx, { 5, 4, 1, 2, 2 }, format::bfyx, { 80, 6, 1, 1 }, 5, 5, 4, data_types::f16, format::bfyx
// #define CASE_GATHER_ND_FP16_6D_4 data_types::f16, format::bfwzyx, { 5, 4, 6, 3, 2, 2 }, format::bfwzyx, { 5, 4, 1, 3, 2, 2 }, format::bfyx, { 240, 1, 1, 1 }, 5, 6, 5, data_types::f16, format::bfyx

// #define CASE_GATHER_ND_FP32_4D_1 data_types::f32, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 3, 1, 1, 1 }, format::bfyx, { 3, 7, 9, 8 }, 0, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_4D_2 data_types::f32, format::bfyx, { 6, 7, 9, 8 }, format::bfyx, { 6, 1, 1, 1 }, format::bfyx, { 6, 8, 1, 9 }, 1, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_4D_3 data_types::f32, format::bfyx, { 5, 4, 7, 2 }, format::bfyx, { 5, 4, 1, 2 }, format::bfyx, { 40, 1, 1, 1 }, 3, data_types::f32, format::bfyx

// #define CASE_GATHER_ND_FP32_5D_1 data_types::f32, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfzyx, { 5, 6, 7, 8, 5 }, 0, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_5D_2 data_types::f32, format::bfzyx, { 5, 6, 7, 8, 5 }, format::bfyx, { 5, 1, 1, 1 }, format::bfyx, { 5, 5, 7, 8 }, 1, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_5D_3 data_types::f32, format::bfzyx, { 5, 4, 7, 8, 5 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 20, 1, 1, 1 }, 2, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_5D_4 data_types::f32, format::bfzyx, { 5, 4, 7, 8, 3 }, format::bfyx, { 5, 4, 1, 3 }, format::bfyx, { 60, 7, 1, 1 }, 3, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_5D_5 data_types::f32, format::bfzyx, { 5, 4, 7, 2, 3 }, format::bfzyx, { 5, 4, 1, 2, 3 }, format::bfyx, { 120, 1, 1, 1 }, 4, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_5D_6 data_types::f32, format::bfzyx, { 5, 4, 7, 4, 4 }, format::bfzyx, { 5, 4, 1, 1, 3 }, format::bfzyx, { 20, 3, 7, 4, 1 }, 2, data_types::f32, format::bfyx

// #define CASE_GATHER_ND_FP32_6D_1 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 8, 5 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 20, 2, 6, 7 }, 5, 4, 2, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_6D_2 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 8, 2 }, format::bfyx, { 5, 4, 2, 2 }, format::bfyx, { 40, 6, 1, 1 }, 5, 4, 3, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_6D_3 data_types::f32, format::bfwzyx, { 5, 4, 6, 7, 2, 2 }, format::bfzyx, { 5, 4, 1, 2, 2 }, format::bfyx, { 80, 6, 1, 1 }, 5, 5, 4, data_types::f32, format::bfyx
// #define CASE_GATHER_ND_FP32_6D_4 data_types::f32, format::bfwzyx, { 5, 4, 6, 3, 2, 2 }, format::bfwzyx, { 5, 4, 1, 3, 2, 2 }, format::bfyx, { 240, 1, 1, 1 }, 5, 6, 5, data_types::f32, format::bfyx

class gather_elements_quantize : public GatherElementsPrimitiveFusingTest {};
TEST_P(gather_elements_quantize, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_elements_indices", get_mem(get_indices_layout(p), 0, p.output_shape.sizes()[p.axis])),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather_elements("gather_elements_prim", "input", "gather_elements_indices", p.axis),
        quantize("quantize", "gather_elements_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_elements_quantize, ::testing::ValuesIn(std::vector<gather_elements_test_params>{
    gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 3 },

    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 3 },

    // gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 3 },

    // gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 3 },

    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 3 },

    // gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 3 },
    // gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 3 },
}));

class gather_elements_activation_scale_eltwise : public GatherElementsPrimitiveFusingTest {};
TEST_P(gather_elements_activation_scale_eltwise, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_elements_indices", get_mem(get_indices_layout(p), 0, p.axis - 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gather_elements("gather_elements", "input", "gather_elements_indices", p.axis),
        activation("activation", "gather_elements", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        eltwise("eltwise", { "scale", "eltwise_data" }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_elements_activation_scale_eltwise, ::testing::ValuesIn(std::vector<gather_elements_test_params>{
    gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 5 },

    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 5 },
    gather_elements_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 5 },

//     gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 5 },

//     gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 5 },

//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 5 },

//     gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 5 },
//     gather_elements_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 5 },
}));
