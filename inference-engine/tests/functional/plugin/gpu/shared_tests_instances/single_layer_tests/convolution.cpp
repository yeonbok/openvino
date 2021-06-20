// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 3}};
const std::vector<std::vector<size_t >> strides = {{1, 1}};
const std::vector<size_t> inputWidths = {{128}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t >> dilations = {{1, 1}};
const std::vector<size_t> numInChannels = {96};
const std::vector<size_t> numOutChannels = {3};
const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
};
const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(inputWidths),
        ::testing::ValuesIn(numInChannels),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPadding_taylor, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ConvolutionLayerTest::getTestCaseName);

}  // namespace
