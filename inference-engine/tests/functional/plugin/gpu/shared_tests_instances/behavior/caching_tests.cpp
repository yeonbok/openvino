// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
    static const std::vector<ngraph::element::Type> precisionsGPU = {
            ngraph::element::f32,
            ngraph::element::f16,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::i8,
            ngraph::element::u8,
            ngraph::element::i16,
            ngraph::element::u16,
    };

    static const std::vector<std::size_t> batchSizesGPU = {
            1, 2
    };

    INSTANTIATE_TEST_CASE_P(smoke_CachingSupportCase_GPU, LoadNetworkCacheTestBase,
                            ::testing::Combine(
                                    ::testing::ValuesIn(LoadNetworkCacheTestBase::getStandardFunctions()),
                                    ::testing::ValuesIn(precisionsGPU),
                                    ::testing::ValuesIn(batchSizesGPU),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                            LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
