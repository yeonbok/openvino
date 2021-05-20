// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for clDNN plugin
 *        To use in SetConfig() method of plugins
 *
 * @file cldnn_config.hpp
 */
#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief GPU plugin configuration
 */
namespace CLDNNConfigParams {

/**
* @brief shortcut for defining configuration keys
*/
#define CLDNN_CONFIG_KEY(name) InferenceEngine::CLDNNConfigParams::_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_KEY(name) DECLARE_CONFIG_KEY(CLDNN_##name)
#define DECLARE_CLDNN_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(CLDNN_##name)

/**
* @brief This key instructs the clDNN plugin to use the OpenCL queue priority hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf
* this option should be used with an unsigned integer value (1 is lowest priority)
* 0 means no priority hint is set and default queue is created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_PRIORITY);

/**
* @brief This key instructs the clDNN plugin to use throttle hints the OpenCL queue throttle hint
* as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
* chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
* 0 means no throttle hint is set and default queue created.
*/
DECLARE_CLDNN_CONFIG_KEY(PLUGIN_THROTTLE);

/**
* @brief This key controls clDNN memory pool optimization.
* Turned off by default.
*/
DECLARE_CLDNN_CONFIG_KEY(MEM_POOL);

/**
* @brief This key defines the directory name to which clDNN graph visualization will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(GRAPH_DUMPS_DIR);

/**
* @brief This key defines the directory name to which full program sources will be dumped.
*/
DECLARE_CLDNN_CONFIG_KEY(SOURCES_DUMPS_DIR);

/**
* @brief This key enables FP16 precision for quantized models.
* By default the model is converted to FP32 precision before running LPT. If this key is enabled (default), then non-quantized layers
* will be converted back to FP16 after LPT, which might imrpove the performance if a model has a lot of compute operations in
* non-quantized path. This key has no effect if current device doesn't have INT8 optimization capabilities.
*/
DECLARE_CLDNN_CONFIG_KEY(ENABLE_FP16_FOR_QUANTIZED_MODELS);

/**
* @brief This key should be set to correctly handle NV12 input without pre-processing.
* Turned off by default.
*/
DECLARE_CLDNN_CONFIG_KEY(NV12_TWO_INPUTS);

/**
* @brief This key sets the max number of host threads that can be used by GPU plugin on model loading.
* Default value is maximum number of threads available in the environment.
*/
DECLARE_CLDNN_CONFIG_KEY(MAX_NUM_THREADS);

/**
* @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration count.
* This key is turned on by default. Note that unrolling loops will cause increase of clDNN build time,
* because the kernels in the body network are created and built for all iterations.
* Also, disabling unrolling for loops with small iteration count and small body network might perform worse than
* unrolling, because the overhead for handling subnetworks might be more significant.
* Thus this config should be set considering the iteration count and body network's size.
*/
DECLARE_CLDNN_CONFIG_KEY(ENABLE_LOOP_UNROLLING);

}  // namespace CLDNNConfigParams
}  // namespace InferenceEngine
