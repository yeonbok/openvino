#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
CNNNetwork cnnNetwork = core.ReadNetwork(FLAGS_m);
std::map<std::string, Parameter> options = {{"CNN_NETWORK", &cnnNetwork}}; // Required. Set the address of the target network.
options.insert(std::make_pair("GPU_THROGHPUT_STREAMS", 2)); // Optional. Set only when you want to estimate max batch size for a specific throughtput streams. Default is 1 or throughtput streams set by SetConfig.
options.insert(std::make_pair("BASE_BATCH_SIZE", 32)); // Optional. Set only when you want to estimate max batch size using a specific batch size. (default is 16).
options.insert(std::make_pair("AVAILABLE_DEVICE_MEM_SIZE", 3221225472)); // Optional. Set only when you want to limit the available device mem size,

auto max_batch_size = core.GetMetric("GPU", METRIC_KEY(MAX_BATCH_SIZE), options).as<unsigned int>();
//! [part0]
}
