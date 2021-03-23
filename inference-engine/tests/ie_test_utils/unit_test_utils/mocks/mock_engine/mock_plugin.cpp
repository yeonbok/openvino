// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <utility>
#include <map>
#include <string>

#include "mock_plugin.hpp"
#include <cpp_interfaces/exception2status.hpp>
#include "description_buffer.hpp"

using namespace std;
using namespace InferenceEngine;

MockPlugin::MockPlugin(InferenceEngine::IInferencePlugin *target) {
    _target = target;
}

void MockPlugin::SetConfig(const std::map<std::string, std::string>& config) {
    this->config = config;
}

Parameter MockPlugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (_target) {
        return _target->GetMetric(name, options);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

ExecutableNetwork
MockPlugin::LoadNetwork(const CNNNetwork &network,
                        const std::map<std::string, std::string> &config) {
    if (_target) {
        return _target->LoadNetwork(network, config);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

ExecutableNetwork
MockPlugin::LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config,
                        RemoteContext::Ptr context) {
    if (_target) {
        return _target->LoadNetwork(network, config, context);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

ExecutableNetworkInternal::Ptr
MockPlugin::LoadExeNetworkImpl(const CNNNetwork& network,
                               const std::map<std::string, std::string>& config) {
    return {};
}

InferenceEngine::ExecutableNetwork
MockPlugin::ImportNetworkImpl(std::istream& networkModel,
                              const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->ImportNetwork(networkModel, config);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

InferenceEngine::ExecutableNetwork
MockPlugin::ImportNetworkImpl(std::istream& networkModel,
                              const InferenceEngine::RemoteContext::Ptr& context,
                              const std::map<std::string, std::string>& config) {
    if (_target) {
        return _target->ImportNetwork(networkModel, context, config);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

InferenceEngine::RemoteContext::Ptr MockPlugin::GetDefaultContext(const InferenceEngine::ParamMap& params) {
    if (_target) {
        return _target->GetDefaultContext(params);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}

InferenceEngine::QueryNetworkResult
MockPlugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                         const std::map<std::string, std::string>& config) const {
    if (_target) {
        return _target->QueryNetwork(network, config);
    } else {
        THROW_IE_EXCEPTION_WITH_STATUS(NotImplemented);
    }
}


InferenceEngine::IInferencePlugin *__target = nullptr;

INFERENCE_PLUGIN_API(void) CreatePluginEngine(std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) {
    IInferencePlugin *p = nullptr;
    std::swap(__target, p);
    plugin = std::make_shared<MockPlugin>(p);
}

INFERENCE_PLUGIN_API(InferenceEngine::IInferencePlugin*)
CreatePluginEngineProxy(InferenceEngine::IInferencePlugin *target) {
    return new MockPlugin(target);
}

INFERENCE_PLUGIN_API(void) InjectProxyEngine(InferenceEngine::IInferencePlugin *target) {
    __target = target;
}
