// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_executable_network.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(ExecutableNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<IExecutableNetwork> nlptr = nullptr;
    ASSERT_THROW(ExecutableNetwork exec(nlptr), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetOutputsInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetInputsInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetInputsInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExport) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::string()), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedExportStream) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.Export(std::cout), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, nothrowsOnUninitializedCast) {
    ExecutableNetwork exec;
    ASSERT_NO_THROW((void)static_cast<IExecutableNetwork::Ptr &>(exec));
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetExecGraphInfo) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetExecGraphInfo(), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedQueryState) {
    IE_SUPPRESS_DEPRECATED_START
    ExecutableNetwork exec;
    ASSERT_THROW(exec.QueryState(), InferenceEngine::Exception);
    IE_SUPPRESS_DEPRECATED_END
}

TEST(ExecutableNetworkTests, throwsOnUninitializedSetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.SetConfig({{}}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetConfig) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetConfig({}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetMetric) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetMetric({}), InferenceEngine::Exception);
}

TEST(ExecutableNetworkTests, throwsOnUninitializedGetContext) {
    ExecutableNetwork exec;
    ASSERT_THROW(exec.GetContext(), InferenceEngine::Exception);
}