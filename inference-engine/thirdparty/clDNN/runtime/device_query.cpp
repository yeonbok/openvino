// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"

#include <map>
#include <string>

namespace cldnn {

device_query::device_query(engine_types engine_type, runtime_types runtime_type, void* user_context, void* user_device, int target_tile_id) {
    switch (engine_type) {
    case engine_types::ocl: {
        if (runtime_type != runtime_types::ocl)
            throw std::runtime_error("Unsupported runtime type for ocl engine");

        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device, target_tile_id);
        break;
    }
    default: throw std::runtime_error("Unsupported engine type in device_query");
    }

    if (_available_devices.empty()) {
        throw std::runtime_error("No suitable devices found for requested engine and runtime types");
    }
}
}  // namespace cldnn
