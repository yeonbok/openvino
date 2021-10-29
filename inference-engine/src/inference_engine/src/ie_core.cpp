// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include <sys/stat.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cnn_network_ngraph_impl.hpp"
#include "compilation_context.hpp"
#include "cpp/ie_cnn_network.h"
#include "cpp/ie_plugin.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "file_utils.h"
#include "ie_cache_guard.hpp"
#include "ie_cache_manager.hpp"
#include "ie_icore.hpp"
#include "ie_itt.hpp"
#include "ie_network_reader.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_remote_context.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine::PluginConfigParams;
using namespace std::placeholders;

namespace ov {
namespace runtime {

namespace {

template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

std::string parseXmlConfig(const std::string& xmlFile) {
    std::string xmlConfigFile_ = xmlFile;
    if (xmlConfigFile_.empty()) {
        // register plugins from default plugins.xml config
        ov::util::FilePath xmlConfigFileDefault =
            FileUtils::makePath(ie::getInferenceEngineLibraryPath(), ov::util::to_file_path("plugins.xml"));
        xmlConfigFile_ = ov::util::from_file_path(xmlConfigFileDefault);
    }
    return xmlConfigFile_;
}

template <typename T = ie::Parameter>
Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[ie::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else if (deviceName.find("AUTO") == 0) {
        deviceName_ = "AUTO";
        if (deviceName.find("AUTO:") == 0) {
            config_[ie::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] =
                deviceName.substr(std::string("AUTO:").size());
        }
    } else {
        if (deviceName_.empty()) {
            deviceName_ = "AUTO";
        }
        ie::DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}

ie::Parameter copyParameterValue(const ie::Parameter& value) {
    if (value.is<bool>()) {
        return {value.as<bool>()};
    } else if (value.is<int>()) {
        return {value.as<int>()};
    } else if (value.is<unsigned int>()) {
        return {value.as<unsigned int>()};
    } else if (value.is<float>()) {
        return {value.as<float>()};
    } else if (value.is<std::string>()) {
        return {value.as<std::string>()};
    } else if (value.is<std::vector<std::string>>()) {
        return {value.as<std::vector<std::string>>()};
    } else if (value.is<std::vector<int>>()) {
        return {value.as<std::vector<int>>()};
    } else if (value.is<std::vector<float>>()) {
        return {value.as<std::vector<float>>()};
    } else if (value.is<std::vector<unsigned int>>()) {
        return {value.as<std::vector<unsigned int>>()};
    } else if (value.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        return {value.as<std::tuple<unsigned int, unsigned int, unsigned int>>()};
    } else if (value.is<std::tuple<unsigned int, unsigned int>>()) {
        return {value.as<std::tuple<unsigned int, unsigned int>>()};
    }

    return std::move(value);
}

template <typename F>
void allowNotImplemented(F&& f) {
    try {
        f();
    } catch (const ie::NotImplemented&) {
    }
}

}  // namespace

class CoreImpl : public ie::ICore, public std::enable_shared_from_this<ie::ICore> {
    mutable std::map<std::string, ov::runtime::InferencePlugin> plugins;

    class CoreConfig final {
    public:
        struct CacheConfig {
            std::string _cacheDir;
            std::shared_ptr<ie::ICacheManager> _cacheManager;
        };

        void setAndUpdate(std::map<std::string, std::string>& config) {
            auto it = config.find(CONFIG_KEY(CACHE_DIR));
            if (it != config.end()) {
                std::lock_guard<std::mutex> lock(_cacheConfigMutex);
                _cacheConfig._cacheDir = it->second;
                if (!it->second.empty()) {
                    FileUtils::createDirectoryRecursive(it->second);
                    _cacheConfig._cacheManager = std::make_shared<ie::FileStorageCacheManager>(std::move(it->second));
                } else {
                    _cacheConfig._cacheManager = nullptr;
                }

                config.erase(it);
            }
        }

        // Creating thread-safe copy of config including shared_ptr to ICacheManager
        CacheConfig getCacheConfig() const {
            std::lock_guard<std::mutex> lock(_cacheConfigMutex);
            return _cacheConfig;
        }

    private:
        mutable std::mutex _cacheConfigMutex;
        CacheConfig _cacheConfig;
    };

    // Core settings (cache config, etc)
    CoreConfig coreConfig;

    ie::CacheGuard cacheGuard;

    struct PluginDescriptor {
        ov::util::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<ov::util::FilePath> listOfExtentions;
    };

    mutable std::unordered_set<std::string> opsetNames;
    mutable std::vector<ie::IExtensionPtr> extensions;

    std::map<std::string, PluginDescriptor> pluginRegistry;
    mutable std::mutex pluginsMutex;  // to lock parallel access to pluginRegistry and plugins

    const bool newAPI;

    bool DeviceSupportsImportExport(const std::string& deviceName) const override {
        auto parsed = parseDeviceNameIntoConfig(deviceName);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        return DeviceSupportsImportExport(plugin);
    }

    bool DeviceSupportsImportExport(const ov::runtime::InferencePlugin& plugin) const {
        std::vector<std::string> supportedMetricKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_METRICS), {});
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(IMPORT_EXPORT_SUPPORT));
        bool supported = (it != supportedMetricKeys.end()) && plugin.get_metric(METRIC_KEY(IMPORT_EXPORT_SUPPORT), {});
        return supported;
    }

    bool DeviceSupportsCacheDir(const ov::runtime::InferencePlugin& plugin) const {
        return DeviceSupportsConfigKey(plugin, CONFIG_KEY(CACHE_DIR));
    }

    bool DeviceSupportsConfigKey(const ov::runtime::InferencePlugin& plugin, const std::string& key) const {
        bool supported = false;
        std::vector<std::string> supportedMetricKeys;
        try {
            // If plugin doesn't support 'SUPPORTED_METRICS' - treat it as config is not supported as well
            supportedMetricKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_METRICS), {}).as<std::vector<std::string>>();
        } catch (...) {
        }
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        if (it != supportedMetricKeys.end()) {
            std::vector<std::string> configKeys = plugin.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
            supported = std::find(configKeys.begin(), configKeys.end(), key) != configKeys.end();
        }
        return supported;
    }

    ov::runtime::SoPtr<ie::IExecutableNetworkInternal> compile_model_impl(
        const InferenceEngine::CNNNetwork& network,
        InferencePlugin& plugin,
        const std::map<std::string, std::string>& parsedConfig,
        const ie::RemoteContext::Ptr& context,
        const std::string& blobID,
        const std::string& modelPath = std::string(),
        bool forceDisableCache = false) {
        OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "CoreImpl::compile_model_impl");
        ov::runtime::SoPtr<ie::IExecutableNetworkInternal> execNetwork;
        execNetwork = context ? plugin.compile_model(network, context, parsedConfig)
                              : plugin.compile_model(network, parsedConfig);
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (!forceDisableCache && cacheManager && DeviceSupportsImportExport(plugin)) {
            try {
                // need to export network for further import from "cache"
                OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Export");
                cacheManager->writeCacheEntry(blobID, [&](std::ostream& networkStream) {
                    networkStream << ie::CompiledBlobHeader(
                        ie::GetInferenceEngineVersion()->buildNumber,
                        ie::NetworkCompilationContext::calculateFileInfo(modelPath));
                    execNetwork->Export(networkStream);
                });
            } catch (...) {
                cacheManager->removeCacheEntry(blobID);
                throw;
            }
        }
        return execNetwork;
    }

    ov::runtime::SoPtr<ie::IExecutableNetworkInternal> LoadNetworkFromCache(
        const std::shared_ptr<ie::ICacheManager>& cacheManager,
        const std::string& blobId,
        ov::runtime::InferencePlugin& plugin,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<ie::RemoteContext>& context,
        bool& networkIsImported,
        const std::string& modelPath = std::string()) {
        ov::runtime::SoPtr<ie::IExecutableNetworkInternal> execNetwork;
        struct HeaderException {};

        OPENVINO_ASSERT(cacheManager != nullptr);
        try {
            cacheManager->readCacheEntry(blobId, [&](std::istream& networkStream) {
                OV_ITT_SCOPE(FIRST_INFERENCE,
                             ie::itt::domains::IE_LT,
                             "Core::LoadNetworkFromCache::ReadStreamAndImport");
                try {
                    ie::CompiledBlobHeader header;
                    networkStream >> header;
                    if (header.getIeVersion() != ie::GetInferenceEngineVersion()->buildNumber) {
                        // Build number mismatch, don't use this cache
                        throw ie::NetworkNotRead("Version does not match");
                    }
                    if (header.getFileInfo() != ie::NetworkCompilationContext::calculateFileInfo(modelPath)) {
                        // Original file is changed, don't use cache
                        throw ie::NetworkNotRead("Original model file is changed");
                    }
                } catch (...) {
                    throw HeaderException();
                }

                execNetwork = context ? plugin.import_model(networkStream, context, config)
                                      : plugin.import_model(networkStream, config);
                networkIsImported = true;
            });
        } catch (const HeaderException&) {
            // For these exceptions just remove old cache and set that import didn't work
            cacheManager->removeCacheEntry(blobId);
            networkIsImported = false;
        } catch (...) {
            cacheManager->removeCacheEntry(blobId);
            networkIsImported = false;
            // TODO: temporary disabled by #54335. In future don't throw only for new 'blob_outdated' exception
            // throw;
        }
        return execNetwork;
    }

    std::map<std::string, std::string> CreateCompileConfig(const ov::runtime::InferencePlugin& plugin,
                                                           const std::string& deviceFamily,
                                                           const std::map<std::string, std::string>& origConfig) const {
        std::map<std::string, ie::Parameter> getMetricConfig;
        auto compileConfig = origConfig;

        // 0. Remove TARGET_FALLBACK key, move it to getMetricConfig
        auto targetFallbackIt = compileConfig.find("TARGET_FALLBACK");
        if (targetFallbackIt != compileConfig.end()) {
            getMetricConfig[targetFallbackIt->first] = targetFallbackIt->second;
            compileConfig.erase(targetFallbackIt);
        }

        // 1. remove DEVICE_ID key
        auto deviceIt = compileConfig.find(CONFIG_KEY(DEVICE_ID));
        if (deviceIt != compileConfig.end()) {
            getMetricConfig[deviceIt->first] = deviceIt->second;
            compileConfig.erase(deviceIt);
        }

        // 2. replace it with DEVICE_ARCHITECTURE value
        std::vector<std::string> supportedMetricKeys =
            plugin.get_metric(METRIC_KEY(SUPPORTED_METRICS), getMetricConfig);
        auto archIt =
            std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(DEVICE_ARCHITECTURE));
        if (archIt != supportedMetricKeys.end()) {
            auto value = plugin.get_metric(METRIC_KEY(DEVICE_ARCHITECTURE), getMetricConfig);
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = value.as<std::string>();
        } else {
            // Take device name if device does not support DEVICE_ARCHITECTURE metric
            compileConfig[METRIC_KEY(DEVICE_ARCHITECTURE)] = deviceFamily;
        }
        return compileConfig;
    }

    std::string CalculateNetworkHash(const ie::CNNNetwork& network,
                                     const std::string& deviceFamily,
                                     const ov::runtime::InferencePlugin& plugin,
                                     const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return ie::NetworkCompilationContext::computeHash(network, compileConfig);
    }

    std::string CalculateFileHash(const std::string& modelName,
                                  const std::string& deviceFamily,
                                  const ov::runtime::InferencePlugin& plugin,
                                  const std::map<std::string, std::string>& config) const {
        auto compileConfig = CreateCompileConfig(plugin, deviceFamily, config);
        return ie::NetworkCompilationContext::computeHash(modelName, compileConfig);
    }

public:
    CoreImpl(bool _newAPI) : newAPI(_newAPI) {
        opsetNames.insert("opset1");
        opsetNames.insert("opset2");
        opsetNames.insert("opset3");
        opsetNames.insert("opset4");
        opsetNames.insert("opset5");
        opsetNames.insert("opset6");
        opsetNames.insert("opset7");
    }

    ~CoreImpl() override = default;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file. The function supports UNICODE
     * path
     * @param xmlConfigFile An .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto parse_result = ParseXml(xmlConfigFile.c_str());
        if (!parse_result.error_msg.empty()) {
            IE_THROW() << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        FOREACH_CHILD (pluginNode, devicesNode, "plugin") {
            std::string deviceName = GetStrAttr(pluginNode, "name");
            ov::util::FilePath pluginPath = ov::util::to_file_path(GetStrAttr(pluginNode, "location").c_str());

            if (deviceName.find('.') != std::string::npos) {
                IE_THROW() << "Device name must not contain dot '.' symbol";
            }

            // append IR library path for default IE plugins
            {
                ov::util::FilePath absFilePath = FileUtils::makePath(ie::getInferenceEngineLibraryPath(), pluginPath);
                if (FileUtils::fileExist(absFilePath))
                    pluginPath = absFilePath;
            }

            // check properties
            auto propertiesNode = pluginNode.child("properties");
            std::map<std::string, std::string> config;

            if (propertiesNode) {
                FOREACH_CHILD (propertyNode, propertiesNode, "property") {
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::vector<ov::util::FilePath> listOfExtentions;

            if (extensionsNode) {
                FOREACH_CHILD (extensionNode, extensionsNode, "extension") {
                    ov::util::FilePath extensionLocation =
                        ov::util::to_file_path(GetStrAttr(extensionNode, "location").c_str());
                    listOfExtentions.push_back(extensionLocation);
                }
            }

            // fill value in plugin registry for later lazy initialization
            {
                PluginDescriptor desc = {pluginPath, config, listOfExtentions};
                pluginRegistry[deviceName] = desc;
            }
        }
    }

    //
    // ICore public API
    //

    /**
     * @brief Returns global task executor
     * @return Reference to task executor
     */
    ie::ITaskExecutor::Ptr GetTaskExecutor() const override {
        return nullptr;
    }

    ie::CNNNetwork ReadNetwork(const std::string& modelPath, const std::string& binPath) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from file");
        return InferenceEngine::details::ReadNetwork(modelPath, binPath, extensions, newAPI);
    }

    ie::CNNNetwork ReadNetwork(const std::string& model, const ie::Blob::CPtr& weights) const override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::IE_RT, "CoreImpl::ReadNetwork from memory");
        return InferenceEngine::details::ReadNetwork(model, weights, extensions, newAPI);
    }

    bool isNewAPI() const override {
        return newAPI;
    }

    // TODO: In future this method can be added to ICore interface
    ov::runtime::SoPtr<ie::IExecutableNetworkInternal> LoadNetwork(const ie::CNNNetwork& network,
                                                                   const std::shared_ptr<ie::RemoteContext>& context,
                                                                   const std::map<std::string, std::string>& config) {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::RemoteContext");
        if (context == nullptr) {
            IE_THROW() << "Remote context is null";
        }
        auto parsed = parseDeviceNameIntoConfig(context->getDeviceName(), config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        ov::runtime::SoPtr<ie::IExecutableNetworkInternal> res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            auto hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            bool loadedFromCache = false;
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, context, loadedFromCache);
            if (!loadedFromCache) {
                res = compile_model_impl(network, plugin, parsed._config, context, hash);
            }
        } else {
            res = compile_model_impl(network, plugin, parsed._config, context, {});
        }
        return res;
    }

    ie::SoExecutableNetworkInternal LoadNetwork(const ie::CNNNetwork& network,
                                                const std::string& deviceName,
                                                const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::CNN");
        bool forceDisableCache = config.count(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE)) > 0;
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        if (forceDisableCache) {
            // remove this config key from parsed as plugins can throw unsupported exception
            parsed._config.erase(CONFIG_KEY_INTERNAL(FORCE_DISABLE_CACHE));
        }
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        ov::runtime::SoPtr<ie::IExecutableNetworkInternal> res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (!forceDisableCache && cacheManager && DeviceSupportsImportExport(plugin)) {
            auto hash = CalculateNetworkHash(network, parsed._deviceName, plugin, parsed._config);
            bool loadedFromCache = false;
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, nullptr, loadedFromCache);
            if (!loadedFromCache) {
                res = compile_model_impl(network, plugin, parsed._config, nullptr, hash, {}, forceDisableCache);
            }
        } else {
            res = compile_model_impl(network, plugin, parsed._config, nullptr, {}, {}, forceDisableCache);
        }
        return {{res._so}, res._ptr};
    }

    ie::SoExecutableNetworkInternal LoadNetwork(const std::string& modelPath,
                                                const std::string& deviceName,
                                                const std::map<std::string, std::string>& config) override {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "Core::LoadNetwork::Path");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto plugin = GetCPPPluginByName(parsed._deviceName);
        ov::runtime::SoPtr<ie::IExecutableNetworkInternal> res;
        auto cacheManager = coreConfig.getCacheConfig()._cacheManager;
        if (cacheManager && DeviceSupportsImportExport(plugin)) {
            bool loadedFromCache = false;
            auto hash = CalculateFileHash(modelPath, parsed._deviceName, plugin, parsed._config);
            auto lock = cacheGuard.getHashLock(hash);
            res = LoadNetworkFromCache(cacheManager, hash, plugin, parsed._config, nullptr, loadedFromCache, modelPath);
            if (!loadedFromCache) {
                auto cnnNetwork = ReadNetwork(modelPath, std::string());
                res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, hash, modelPath);
            }
        } else if (cacheManager) {
            res = plugin.compile_model(modelPath, parsed._config);
        } else {
            auto cnnNetwork = ReadNetwork(modelPath, std::string());
            res = compile_model_impl(cnnNetwork, plugin, parsed._config, nullptr, {}, modelPath);
        }
        return {{res._so}, res._ptr};
    }

    ie::SoExecutableNetworkInternal ImportNetwork(std::istream& networkModel,
                                                  const std::string& deviceName,
                                                  const std::map<std::string, std::string>& config) override {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto exec = GetCPPPluginByName(parsed._deviceName).import_model(networkModel, parsed._config);

        if (isNewAPI()) {
            // create getInputs() based on GetInputsInfo()
            using namespace InferenceEngine::details;

            if (exec->getInputs().empty()) {
                const auto& inputsInfo = exec->GetInputsInfo();
                OPENVINO_ASSERT(!inputsInfo.empty(), "inputsInfo is empty after network import");

                std::vector<std::shared_ptr<const ov::Node>> params;
                params.reserve(inputsInfo.size());
                for (auto&& input : inputsInfo) {
                    auto param = std::make_shared<ov::op::v0::Parameter>(
                        convertPrecision(input.second->getPrecision()),
                        ov::PartialShape(input.second->getTensorDesc().getDims()));
                    param->set_friendly_name(input.first);
                    param->get_output_tensor(0).add_names({input.first});
                    params.emplace_back(std::move(param));
                }

                exec->setInputs(params);
            }

            if (exec->getOutputs().empty()) {
                const auto& outputsInfo = exec->GetOutputsInfo();
                OPENVINO_ASSERT(!outputsInfo.empty(), "outputsInfo is empty after network import");

                std::vector<std::shared_ptr<const ov::Node>> results;
                results.reserve(outputsInfo.size());
                for (auto&& output : outputsInfo) {
                    auto fake_param = std::make_shared<ov::op::v0::Parameter>(
                        convertPrecision(output.second->getPrecision()),
                        ov::PartialShape(output.second->getTensorDesc().getDims()));
                    fake_param->set_friendly_name(output.first);
                    auto result = std::make_shared<ov::op::v0::Result>(fake_param);
                    result->get_output_tensor(0).add_names({output.first});
                    results.emplace_back(std::move(result));
                }
                exec->setOutputs(results);
            }

            // but for true support plugins need:
            // 1. ensure order or parameters and results as in ov::Function
            // 2. provide tensor names for inputs and outputs
            // 3. precisions for getInputs and getOutputs should be taken from GetInputsInfo / GetOutputsInfo
            //    not from ngraph. Plugins should use SetExeNetworkInfo
        }

        return {{exec._so}, exec._ptr};
    }

    ie::QueryNetworkResult QueryNetwork(const ie::CNNNetwork& network,
                                        const std::string& deviceName,
                                        const std::map<std::string, std::string>& config) const override {
        OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::QueryNetwork");
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        auto res = GetCPPPluginByName(parsed._deviceName).query_model(network, parsed._config);
        if (!network.getFunction() || res.supportedLayersMap.empty())
            return res;

        const auto& func = network.getFunction();
        auto specialized_function = ngraph::clone_function(*func);

        std::string defDevice = res.supportedLayersMap.begin()->second;
        ngraph::pass::ConstantFolding().run_on_function(specialized_function);
        std::unordered_set<std::string> opNames;

        for (const auto& op : specialized_function->get_ops())
            opNames.emplace(op->get_friendly_name());

        for (const auto& op : func->get_ops()) {
            if (opNames.find(op->get_friendly_name()) == opNames.end() ||
                (!res.supportedLayersMap.count(op->get_friendly_name()) &&
                 std::dynamic_pointer_cast<ngraph::op::Constant>(op)))
                res.supportedLayersMap[op->get_friendly_name()] = defDevice;
        }
        return res;
    }

    ie::Parameter GetMetric(const std::string& deviceName,
                            const std::string& name,
                            const ie::ParamMap& options = {}) const override {
        // HETERO case
        {
            if (deviceName.find("HETERO:") == 0) {
                IE_THROW()
                    << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        // MULTI case
        {
            if (deviceName.find("MULTI:") == 0) {
                IE_THROW()
                    << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        // AUTO case
        {
            if (deviceName.find("AUTO:") == 0) {
                IE_THROW()
                    << "You can get specific metrics with the GetMetric only for the AUTO itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
            }
        }

        auto parsed = parseDeviceNameIntoConfig(deviceName);
        for (auto o : options) {
            parsed._config.insert(o);
        }

        // we need to return a copy of Parameter object which is created on Core side,
        // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(GetCPPPluginByName(parsed._deviceName).get_metric(name, parsed._config));
    }

    ie::Parameter GetConfig(const std::string& deviceName, const std::string& name) const override {
        auto parsed = parseDeviceNameIntoConfig(deviceName);

        // we need to return a copy of Parameter object which is created on Core side,
        // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config));
    }

    /**
     * @brief Returns devices available for neural networks inference
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, MYRIAD }
     * If there more than one device of specific type, they are enumerated with .# suffix.
     */
    std::vector<std::string> GetAvailableDevices() const override {
        std::vector<std::string> devices;
        const std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

        for (auto&& deviceName : GetListOfDevicesInRegistry()) {
            std::vector<std::string> devicesIDs;
            try {
                const ie::Parameter p = GetMetric(deviceName, propertyName);
                devicesIDs = p.as<std::vector<std::string>>();
            } catch (ie::Exception&) {
                // plugin is not created by e.g. invalid env
            } catch (ov::Exception&) {
                // plugin is not created by e.g. invalid env
            } catch (std::runtime_error&) {
                // plugin is not created by e.g. invalid env
            } catch (const std::exception& ex) {
                IE_THROW() << "An exception is thrown while trying to create the " << deviceName
                           << " device and call GetMetric: " << ex.what();
            } catch (...) {
                IE_THROW() << "Unknown exception is thrown while trying to create the " << deviceName
                           << " device and call GetMetric";
            }

            if (devicesIDs.size() > 1) {
                for (auto&& deviceID : devicesIDs) {
                    devices.push_back(deviceName + '.' + deviceID);
                }
            } else if (!devicesIDs.empty()) {
                devices.push_back(deviceName);
            }
        }

        return devices;
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName A name of device
     * @return Reference to a CPP plugin wrapper
     */
    ov::runtime::InferencePlugin GetCPPPluginByName(const std::string& pluginName) const {
        OV_ITT_SCOPE(FIRST_INFERENCE, ie::itt::domains::IE_LT, "CoreImpl::GetCPPPluginByName");

        std::lock_guard<std::mutex> lock(pluginsMutex);
        auto deviceName = pluginName;
        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create
        auto it_plugin = plugins.find(deviceName);
        if (it_plugin == plugins.end()) {
            PluginDescriptor desc = it->second;
            auto so = ov::util::load_shared_object(desc.libraryLocation.c_str());
            try {
                using CreateF = void(std::shared_ptr<ie::IInferencePlugin>&);
                std::shared_ptr<ie::IInferencePlugin> plugin_impl;
                reinterpret_cast<CreateF*>(ov::util::get_symbol(so, OV_PP_TOSTRING(IE_CREATE_PLUGIN)))(plugin_impl);
                auto plugin = InferencePlugin{so, plugin_impl};

                {
                    plugin.set_name(deviceName);

                    // Set Inference Engine class reference to plugins
                    std::weak_ptr<ie::ICore> mutableCore = std::const_pointer_cast<ie::ICore>(shared_from_this());
                    plugin.set_core(mutableCore);
                }

                // Add registered extensions to new plugin
                allowNotImplemented([&]() {
                    for (const auto& ext : extensions) {
                        plugin.add_extension(ext);
                    }
                });

                // configuring
                {
                    if (DeviceSupportsCacheDir(plugin)) {
                        auto cacheConfig = coreConfig.getCacheConfig();
                        if (cacheConfig._cacheManager) {
                            desc.defaultConfig[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                        }
                    }
                    allowNotImplemented([&]() {
                        // Add device specific value to support device_name.device_id cases
                        std::vector<std::string> supportedConfigKeys =
                            plugin.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
                        auto config_iter = std::find(supportedConfigKeys.begin(),
                                                     supportedConfigKeys.end(),
                                                     CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
                        const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
                        const std::string deviceKey =
                            supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

                        for (auto pluginDesc : pluginRegistry) {
                            InferenceEngine::DeviceIDParser parser(pluginDesc.first);
                            if (pluginDesc.first.find(deviceName) != std::string::npos &&
                                !parser.getDeviceID().empty()) {
                                pluginDesc.second.defaultConfig[deviceKey] = parser.getDeviceID();
                                plugin.set_config(pluginDesc.second.defaultConfig);
                            }
                        }
                        plugin.set_config(desc.defaultConfig);
                    });

                    allowNotImplemented([&]() {
                        for (auto&& extensionLocation : desc.listOfExtentions) {
                            plugin.add_extension(std::make_shared<ie::Extension>(extensionLocation));
                        }
                    });
                }

                auto result = plugins.emplace(deviceName, plugin).first->second;

                TryToRegisterLibraryAsExtensionUnsafe(desc.libraryLocation);

                return result;
            } catch (const ie::Exception& ex) {
                IE_THROW() << "Failed to create plugin " << ov::util::from_file_path(desc.libraryLocation)
                           << " for device " << deviceName << "\n"
                           << "Please, check your environment\n"
                           << ex.what() << "\n";
            }
        } else {
            return it_plugin->second;
        };
    }

    /**
     * @brief Unload plugin for specified device, but plugin meta-data is still in plugin registry
     * @param deviceName A name of device
     */
    void UnloadPluginByName(const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin meta-data in registry for specified device
     * @param deviceName A name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
            IE_THROW() << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
            IE_THROW() << "Device name must not contain dot '.' symbol";
        }

        // append IR library path for default IE plugins
        ov::util::FilePath pluginPath;
        {
            pluginPath = FileUtils::makePluginLibraryName({}, ov::util::to_file_path(pluginName.c_str()));

            ov::util::FilePath absFilePath = FileUtils::makePath(ie::getInferenceEngineLibraryPath(), pluginPath);
            if (FileUtils::fileExist(absFilePath))
                pluginPath = absFilePath;
        }

        PluginDescriptor desc = {pluginPath, {}, {}};
        pluginRegistry[deviceName] = desc;
    }

    /**
     * @brief Provides a list of plugin names in registry; physically such plugins may not be created
     * @return A list of plugin names
     */
    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::lock_guard<std::mutex> lock(pluginsMutex);

        std::vector<std::string> listOfDevices;
        for (auto&& pluginDesc : pluginRegistry) {
            listOfDevices.push_back(pluginDesc.first);
        }

        return listOfDevices;
    }

    /**
     * @brief Sets config values for a plugin or set of plugins
     * @param deviceName A device name to set config to
     *        If empty, config is set for all the plugins / plugin's meta-data
     * @note  `deviceName` is not allowed in form of MULTI:CPU, HETERO:GPU,CPU, AUTO:CPU
     *        just simple forms like CPU, GPU, MULTI, GPU.0, etc
     */
    void SetConfigForPlugins(const std::map<std::string, std::string>& configMap, const std::string& deviceName) {
        auto config = configMap;

        InferenceEngine::DeviceIDParser parser(deviceName);
        std::string clearDeviceName = parser.getDeviceName();

        std::lock_guard<std::mutex> lock(pluginsMutex);

        if (deviceName.empty()) {
            coreConfig.setAndUpdate(config);
        }

        auto base_desc = pluginRegistry.find(clearDeviceName);
        if (pluginRegistry.find(deviceName) == pluginRegistry.end() && base_desc != pluginRegistry.end()) {
            PluginDescriptor desc = {base_desc->second.libraryLocation, config, base_desc->second.listOfExtentions};
            pluginRegistry[deviceName] = desc;
        }

        // set config for plugins in registry
        bool configIsSet = false;
        for (auto& desc : pluginRegistry) {
            if (deviceName.empty() || deviceName == desc.first) {
                for (auto&& conf : config) {
                    desc.second.defaultConfig[conf.first] = conf.second;
                }
                configIsSet = true;
            }
        }

        if (!configIsSet && !deviceName.empty()) {
            IE_THROW() << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
            if (deviceName.empty() || clearDeviceName == plugin.first) {
                allowNotImplemented([&]() {
                    auto configCopy = config;
                    if (DeviceSupportsCacheDir(plugin.second)) {
                        auto cacheConfig = coreConfig.getCacheConfig();
                        if (cacheConfig._cacheManager) {
                            configCopy[CONFIG_KEY(CACHE_DIR)] = cacheConfig._cacheDir;
                        }
                    }
                    // Add device specific value to support device_name.device_id cases
                    std::vector<std::string> supportedConfigKeys =
                        plugin.second.get_metric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
                    auto config_iter = std::find(supportedConfigKeys.begin(),
                                                 supportedConfigKeys.end(),
                                                 CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID));
                    const bool supportsConfigDeviceID = config_iter != supportedConfigKeys.end();
                    const std::string deviceKey =
                        supportsConfigDeviceID ? CONFIG_KEY_INTERNAL(CONFIG_DEVICE_ID) : CONFIG_KEY(DEVICE_ID);

                    if (!parser.getDeviceID().empty()) {
                        configCopy[deviceKey] = parser.getDeviceID();
                    }
                    plugin.second.set_config(configCopy);
                });
            }
        }
    }

    /**
     * @brief Registers the extension in a Core object
     *        Such extensions can be used for both CNNNetwork readers and device plugins
     */
    void AddExtension(const ie::IExtensionPtr& extension) {
        std::lock_guard<std::mutex> lock(pluginsMutex);
        AddExtensionUnsafe(extension);
    }

    /**
     * @brief Provides a list of extensions
     * @return A list of registered extensions
     */
    const std::vector<ie::IExtensionPtr>& GetExtensions() const {
        return extensions;
    }

    std::map<std::string, ie::Version> GetVersions(const std::string& deviceName) const {
        std::map<std::string, ie::Version> versions;
        std::vector<std::string> deviceNames;

        {
            // for compatibility with samples / demo
            if (deviceName.find("HETERO") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getHeteroDevices(deviceName.substr(pos + 1));
                }
                deviceNames.push_back("HETERO");
            } else if (deviceName.find("MULTI") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
                }
                deviceNames.push_back("MULTI");
            } else if (deviceName.find("AUTO") == 0) {
                auto pos = deviceName.find_first_of(":");
                if (pos != std::string::npos) {
                    deviceNames = ie::DeviceIDParser::getMultiDevices(deviceName.substr(pos + 1));
                }
                deviceNames.emplace_back("AUTO");
            } else {
                deviceNames.push_back(deviceName);
            }
        }

        for (auto&& deviceName_ : deviceNames) {
            ie::DeviceIDParser parser(deviceName_);
            std::string deviceNameLocal = parser.getDeviceName();

            ov::runtime::InferencePlugin cppPlugin = GetCPPPluginByName(deviceNameLocal);
            const ie::Version version = cppPlugin.get_version();
            versions[deviceNameLocal] = version;
        }

        return versions;
    }

private:
    void AddExtensionUnsafe(const ie::IExtensionPtr& extension) const {
        std::map<std::string, ngraph::OpSet> opsets = extension->getOpSets();
        for (const auto& it : opsets) {
            if (opsetNames.find(it.first) != opsetNames.end())
                IE_THROW() << "Cannot add opset with name: " << it.first
                           << ". Opset with the same name already exists.";
            opsetNames.insert(it.first);
        }

        // add extensions for already created plugins
        for (auto& plugin : plugins) {
            try {
                plugin.second.add_extension(extension);
            } catch (...) {
            }
        }
        extensions.emplace_back(extension);
    }

    template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
    void TryToRegisterLibraryAsExtensionUnsafe(const std::basic_string<C>& path) const {
        try {
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>(path);
            AddExtensionUnsafe(extension_ptr);
        } catch (const InferenceEngine::NotFound&) {
        } catch (const InferenceEngine::GeneralError&) {
        }
    }
};

}  // namespace runtime
}  // namespace ov

namespace InferenceEngine {

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        deviceName = deviceNameWithID.substr(0, pos);
        deviceID = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

std::string DeviceIDParser::getDeviceID() const {
    return deviceID;
}

std::string DeviceIDParser::getDeviceName() const {
    return deviceName;
}

std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty())
        deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::vector<std::string> deviceNames;
    auto trim_request_info = [](std::string device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
        auto d = devicesList.substr(0, pos);
        deviceNames.push_back(trim_request_info(d));
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty())
        deviceNames.push_back(trim_request_info(devicesList));

    return deviceNames;
}

class Core::Impl : public ov::runtime::CoreImpl {
public:
    Impl() : ov::runtime::CoreImpl(false) {}
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

    RegisterPlugins(ov::runtime::parseXmlConfig(xmlConfigFile));
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    return _impl->GetVersions(deviceName);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

CNNNetwork Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) const {
    return ReadNetwork(ov::util::wstring_to_string(modelPath), ov::util::wstring_to_string(binPath));
}

#endif

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    return _impl->ReadNetwork(modelPath, binPath);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    return _impl->ReadNetwork(model, weights);
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(network, deviceName, config);
    return {exec._so, exec._ptr};
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(network, std::dynamic_pointer_cast<RemoteContext>(context), config);
    return {{exec._so}, exec._ptr};
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto exec = _impl->LoadNetwork(modelPath, deviceName, config);
    return {exec._so, exec._ptr};
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    if (deviceName.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support remote context";
    }
    if (deviceName.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support remote context";
    }

    auto parsed = ov::runtime::parseDeviceNameIntoConfig(deviceName, params);
    return _impl->GetCPPPluginByName(parsed._deviceName).create_context(parsed._config)._ptr;
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName) {
    if (deviceName.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support remote context";
    }
    if (deviceName.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support remote context";
    }

    auto parsed = ov::runtime::parseDeviceNameIntoConfig(deviceName, ParamMap());
    return _impl->GetCPPPluginByName(parsed._deviceName).get_default_context(parsed._config)._ptr;
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support extensions. Please, set extensions directly to fallback devices";
    }

    _impl->AddExtension(extension);
}

void Core::AddExtension(const IExtensionPtr& extension) {
    _impl->AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");
    auto parsed = ov::runtime::parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(parsed._deviceName).import_model(modelFileName, parsed._config);
    return {{exec._so}, exec._ptr};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");
    auto exec = _impl->ImportNetwork(networkModel, deviceName, config);
    return {exec._so, exec._ptr};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

    std::string deviceName;
    ExportMagic magic = {};
    auto currentPos = networkModel.tellg();
    networkModel.read(magic.data(), magic.size());
    if (exportMagic == magic) {
        std::getline(networkModel, deviceName);
    } else {
        IE_THROW() << "Passed compiled stream does not contain device name. "
                      "Please, provide device name manually";
    }
    networkModel.seekg(currentPos, networkModel.beg);

    auto exec = _impl->GetCPPPluginByName(deviceName).import_model(networkModel, {});
    return {{exec._so}, exec._ptr};
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::ImportNetwork");

    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    auto parsed = ov::runtime::parseDeviceNameIntoConfig(deviceName, config);
    auto exec = _impl->GetCPPPluginByName(deviceName)
                    .import_model(networkModel, std::dynamic_pointer_cast<RemoteContext>(context), parsed._config);
    return {{exec._so}, exec._ptr};
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    return _impl->QueryNetwork(network, deviceName, config);
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    if (deviceName.find("HETERO:") == 0) {
        IE_THROW() << "SetConfig is supported only for HETERO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the HETERO on top.";
    }

    // MULTI case
    if (deviceName.find("MULTI:") == 0) {
        IE_THROW() << "SetConfig is supported only for MULTI itself (without devices). "
                      "You can configure the devices with SetConfig before creating the MULTI on top.";
    }

    // AUTO case
    if (deviceName.find("AUTO:") == 0) {
        IE_THROW() << "SetConfig is supported only for AUTO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the AUTO on top.";
    }

    if (deviceName.empty()) {
        _impl->SetConfigForPlugins(config, std::string());
    } else {
        _impl->SetConfigForPlugins(config, deviceName);
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW() << "You can only GetConfig of the HETERO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW() << "You can only GetConfig of the MULTI itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }
    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW() << "You can only GetConfig of the AUTO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the AUTO on top.";
        }
    }

    auto parsed = ov::runtime::parseDeviceNameIntoConfig(deviceName);

    // we need to return a copy of Parameter object which is created on Core side,
    // not in InferenceEngine plugin side, which can be unloaded from Core in a parallel thread
    // TODO: remove this WA after *-31417 is resolved
    return ov::runtime::copyParameterValue(
        _impl->GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config));
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name, const ParamMap& options) const {
    return _impl->GetMetric(deviceName, name, options);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    return _impl->GetAvailableDevices();
}

void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {
    _impl->RegisterPluginByName(pluginName, deviceName);
}

void Core::RegisterPlugins(const std::string& xmlConfigFile) {
    _impl->RegisterPluginsInRegistry(xmlConfigFile);
}

void Core::UnregisterPlugin(const std::string& deviceName_) {
    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    _impl->UnloadPluginByName(deviceName);
}

}  // namespace InferenceEngine

namespace ov {
namespace runtime {

#define OV_CORE_CALL_STATEMENT(...)                     \
    try {                                               \
        __VA_ARGS__;                                    \
    } catch (const std::exception& ex) {                \
        throw ov::Exception(ex.what());                 \
    } catch (...) {                                     \
        OPENVINO_ASSERT(false, "Unexpected exception"); \
    }

class Core::Impl : public CoreImpl {
public:
    Impl() : ov::runtime::CoreImpl(true) {}
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

    OV_CORE_CALL_STATEMENT(register_plugins(parseXmlConfig(xmlConfigFile)));
}

std::map<std::string, Version> Core::get_versions(const std::string& deviceName) const {
    OV_CORE_CALL_STATEMENT({
        std::map<std::string, Version> versions;
        for (auto&& kvp : _impl->GetVersions(deviceName)) {
            versions[kvp.first] = Version{kvp.second.buildNumber, kvp.second.description};
        }
        return versions;
    })
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<ov::Function> Core::read_model(const std::wstring& modelPath, const std::wstring& binPath) const {
    OV_CORE_CALL_STATEMENT(
        return _impl->ReadNetwork(ov::util::wstring_to_string(modelPath), ov::util::wstring_to_string(binPath))
            .getFunction(););
}
#endif

std::shared_ptr<ov::Function> Core::read_model(const std::string& modelPath, const std::string& binPath) const {
    OV_CORE_CALL_STATEMENT(return _impl->ReadNetwork(modelPath, binPath).getFunction(););
}

std::shared_ptr<ov::Function> Core::read_model(const std::string& model, const ov::runtime::Tensor& weights) const {
    InferenceEngine::Blob::Ptr blob;
    if (weights) {
        blob = weights._impl;
    }
    OV_CORE_CALL_STATEMENT(return _impl->ReadNetwork(model, blob).getFunction(););
}

namespace {

ie::CNNNetwork toCNN(const std::shared_ptr<const ngraph::Function>& model) {
    return ie::CNNNetwork(
        std::make_shared<ie::details::CNNNetworkNGraphImpl>(std::const_pointer_cast<ngraph::Function>(model),
                                                            std::vector<ie::IExtensionPtr>{},
                                                            true));
}

}  // namespace

ExecutableNetwork Core::compile_model(const std::shared_ptr<const ov::Function>& model,
                                      const std::string& deviceName,
                                      const ConfigMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(toCNN(model), deviceName, config);
        return {exec._so, exec._ptr};
    });
}

ExecutableNetwork Core::compile_model(const std::string& modelPath,
                                      const std::string& deviceName,
                                      const ConfigMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(modelPath, deviceName, config);
        return {exec._so, exec._ptr};
    });
}

ExecutableNetwork Core::compile_model(const std::shared_ptr<const ov::Function>& model,
                                      const RemoteContext& context,
                                      const ConfigMap& config) {
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->LoadNetwork(toCNN(model), context._impl, config);
        return {exec._so, exec._ptr};
    });
}

void Core::add_extension(const ie::IExtensionPtr& extension) {
    OV_CORE_CALL_STATEMENT(_impl->AddExtension(extension););
}

ExecutableNetwork Core::import_model(std::istream& modelStream,
                                     const std::string& deviceName,
                                     const ConfigMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");
    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->ImportNetwork(modelStream, deviceName, config);
        return {exec._so, exec._ptr};
    });
}

ExecutableNetwork Core::import_model(std::istream& modelStream, const RemoteContext& context, const ConfigMap& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::IE, "Core::import_model");

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

    std::string deviceName;
    ExportMagic magic = {};
    auto currentPos = modelStream.tellg();
    modelStream.read(magic.data(), magic.size());
    if (exportMagic == magic) {
        std::getline(modelStream, deviceName);
    } else {
        OPENVINO_ASSERT(false,
                        "Passed compiled stream does not contain device name. "
                        "Please, provide device name manually");
    }
    modelStream.seekg(currentPos, modelStream.beg);

    OV_CORE_CALL_STATEMENT({
        auto exec = _impl->GetCPPPluginByName(deviceName).import_model(modelStream, {});
        return {exec._so, exec._ptr};
    });
}

SupportedOpsMap Core::query_model(const std::shared_ptr<const ov::Function>& model,
                                  const std::string& deviceName,
                                  const ConfigMap& config) const {
    OV_CORE_CALL_STATEMENT({
        auto qnResult = _impl->QueryNetwork(toCNN(model), deviceName, config);
        return qnResult.supportedLayersMap;
    });
}

void Core::set_config(const ConfigMap& config, const std::string& deviceName) {
    OPENVINO_ASSERT(deviceName.find("HETERO:") != 0,
                    "set_config is supported only for HETERO itself (without devices). "
                    "You can configure the devices with set_config before creating the HETERO on top.");
    OPENVINO_ASSERT(deviceName.find("MULTI:") != 0,
                    "set_config is supported only for MULTI itself (without devices). "
                    "You can configure the devices with set_config before creating the MULTI on top.");
    OPENVINO_ASSERT(deviceName.find("AUTO:") != 0,
                    "set_config is supported only for AUTO itself (without devices). "
                    "You can configure the devices with set_config before creating the AUTO on top.");

    OV_CORE_CALL_STATEMENT({
        if (deviceName.empty()) {
            _impl->SetConfigForPlugins(config, std::string());
        } else {
            _impl->SetConfigForPlugins(config, deviceName);
        }
    });
}

Parameter Core::get_config(const std::string& deviceName, const std::string& name) const {
    OPENVINO_ASSERT(deviceName.find("HETERO:") != 0,
                    "You can only get_config of the HETERO itself (without devices). "
                    "get_config is also possible for the individual devices before creating the HETERO on top.");
    OPENVINO_ASSERT(deviceName.find("MULTI:") != 0,
                    "You can only get_config of the MULTI itself (without devices). "
                    "get_config is also possible for the individual devices before creating the MULTI on top.");
    OPENVINO_ASSERT(deviceName.find("AUTO:") != 0,
                    "You can only get_config of the AUTO itself (without devices). "
                    "get_config is also possible for the individual devices before creating the AUTO on top.");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(deviceName);

        // we need to return a copy of Parameter object which is created on Core side,
        // not in ie plugin side, which can be unloaded from Core in a parallel thread
        // TODO: remove this WA after *-31417 is resolved
        return copyParameterValue(_impl->GetCPPPluginByName(parsed._deviceName).get_config(name, parsed._config));
    });
}

Parameter Core::get_metric(const std::string& deviceName, const std::string& name) const {
    OV_CORE_CALL_STATEMENT(return _impl->GetMetric(deviceName, name););
}

std::vector<std::string> Core::get_available_devices() const {
    OV_CORE_CALL_STATEMENT(return _impl->GetAvailableDevices(););
}

void Core::register_plugin(const std::string& pluginName, const std::string& deviceName) {
    OV_CORE_CALL_STATEMENT(_impl->RegisterPluginByName(pluginName, deviceName););
}

void Core::unload_plugin(const std::string& deviceName) {
    OV_CORE_CALL_STATEMENT({
        ie::DeviceIDParser parser(deviceName);
        std::string devName = parser.getDeviceName();

        _impl->UnloadPluginByName(devName);
    });
}

void Core::register_plugins(const std::string& xmlConfigFile) {
    OV_CORE_CALL_STATEMENT(_impl->RegisterPluginsInRegistry(xmlConfigFile););
}

RemoteContext Core::create_context(const std::string& deviceName, const ParamMap& params) {
    OPENVINO_ASSERT(deviceName.find("HETERO") != 0, "HETERO device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("MULTI") != 0, "MULTI device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("AUTO") != 0, "AUTO device does not support remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(deviceName, params);
        auto remoteContext = _impl->GetCPPPluginByName(parsed._deviceName).create_context(parsed._config);
        return {remoteContext._so, remoteContext._ptr};
    });
}

RemoteContext Core::get_default_context(const std::string& deviceName) {
    OPENVINO_ASSERT(deviceName.find("HETERO") != 0, "HETERO device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("MULTI") != 0, "MULTI device does not support remote context");
    OPENVINO_ASSERT(deviceName.find("AUTO") != 0, "AUTO device does not support remote context");

    OV_CORE_CALL_STATEMENT({
        auto parsed = parseDeviceNameIntoConfig(deviceName, ParamMap());
        auto remoteContext = _impl->GetCPPPluginByName(parsed._deviceName).get_default_context(parsed._config);
        return {remoteContext._so, remoteContext._ptr};
    });
}

}  // namespace runtime
}  // namespace ov
