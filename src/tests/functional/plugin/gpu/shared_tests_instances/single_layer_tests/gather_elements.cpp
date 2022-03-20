// // Copyright (C) 2018-2022 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

#include <vector>

//NOTE:
//TEST_P와 FixtureClass(GatherElementsLayerTest)는 여기에 shared되어있다.
//tensor들은 shape만 지정해주면 그에맞춰 superclass에서 random값을 할당하는듯
//추가적으로 Fusing까지 체크하는것 같다?
#include "single_layer_tests/gather_elements.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset6;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    // InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
    // InferenceEngine::Precision::I64,
};

// -------------------------------- V6 --------------------------------
INSTANTIATE_TEST_SUITE_P(smoke_GatherElements6_set1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{4, 2, 3, 1}, std::vector<size_t>{4, 2, 3, 5}),//두 shape중 하나
        ::testing::Values(std::vector<size_t>{4, 2, 3, 1}, std::vector<size_t>{4, 2, 3, 5}),//두 shape중 하나
        ::testing::Values(3),//axis=3
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);
}  // namespace
