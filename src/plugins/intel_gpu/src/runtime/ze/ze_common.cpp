// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "openvino/core/except.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include "windows.h"
#else
#error "Level Zero is supported on Linux and Windows only"
#endif

namespace cldnn {
namespace ze {

void *find_ze_symbol(const char *symbol) {
#if defined(__linux__)
    void *handle = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
#elif defined(_WIN32)
    HMODULE handle = LoadLibraryA("ze_loader.dll");
#endif
    if (!handle) {
        return nullptr;
    }

#if defined(__linux__)
    void *f = dlsym(handle, symbol);
#elif defined(_WIN32)
    void *f = GetProcAddress(handle, symbol);
#endif
    OPENVINO_ASSERT(f != nullptr);
    return f;
}


template<typename T>
ze_result_t set_kernel_arg_scalar(ze_kernel_handle_t& kernel, uint32_t idx, const T& val) {
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set scalar " << idx << " (" << ov::element::from<T>().get_type_name() << ")" << val << "\n";
    return zeKernelSetArgumentValue(kernel, idx, sizeof(T), &val);
}

ze_result_t set_kernel_arg(ze_kernel_handle_t& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    OPENVINO_ASSERT(memory_capabilities::is_usm_type(mem->get_allocation_type()), "Unsupported alloc type");
    const auto& buf = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_buffer();
    auto mem_type = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_allocation_type();
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg (" << mem_type << ") " << idx
                            << " mem: " << buf.get() << " size: " << mem->size() << std::endl;

    auto ptr = buf.get();
    return zeKernelSetArgumentValue(kernel, idx, sizeof(ptr), &ptr);
}

void set_arguments_impl(ze_kernel_handle_t kernel,
                         const arguments_desc& args,
                         const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;

    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        ze_result_t status = ZE_RESULT_NOT_READY;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
                }
                break;
            case args_t::OUTPUT:
                if (args[i].index < data.outputs.size() && data.outputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
                }
                break;
            case args_t::WEIGHTS:
                status = set_kernel_arg(kernel, i, data.weights);
                break;
            case args_t::BIAS:
                status = set_kernel_arg(kernel, i, data.bias);
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.weights_zero_points);
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.activations_zero_points);
                break;
            case args_t::COMPENSATION:
                status = set_kernel_arg(kernel, i, data.compensation);
                break;
            case args_t::SCALE_TABLE:
                status = set_kernel_arg(kernel, i, data.scale_table);
                break;
            case args_t::SLOPE:
                status = set_kernel_arg(kernel, i, data.slope);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = set_kernel_arg_scalar<uint8_t>(kernel, i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = set_kernel_arg_scalar<uint16_t>(kernel, i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = set_kernel_arg_scalar<uint32_t>(kernel, i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = set_kernel_arg_scalar<uint64_t>(kernel, i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = set_kernel_arg_scalar<int8_t>(kernel, i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = set_kernel_arg_scalar<int16_t>(kernel, i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = set_kernel_arg_scalar<int32_t>(kernel, i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = set_kernel_arg_scalar<int64_t>(kernel, i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = set_kernel_arg_scalar<float>(kernel, i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = set_kernel_arg_scalar<double>(kernel, i, scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::CELL:
                status = set_kernel_arg(kernel, i, data.cell);
                break;
            case args_t::SHAPE_INFO:
                status = set_kernel_arg(kernel, i, data.shape_info);
                break;
            default:
                break;
        }
        if (status != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

static std::pair<size_t, void*> get_buffer_arg(cldnn::memory::cptr mem) {
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(mem->get_allocation_type()), "Unsupported alloc type");
    const auto& buf = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_buffer();
    auto ptr = buf.get();
    return std::make_pair<size_t, void*>(sizeof(ptr), &ptr);

}

template<typename T>
static std::pair<size_t, void*> get_scalar_arg(const T& val) {
    return std::make_pair<size_t, void*>(sizeof(T), const_cast<T*>(&val));

}

std::pair<size_t, void*> get_arguments_impl(const argument_desc& desc, const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;

    switch (desc.t) {
        case args_t::INPUT:
            if (desc.index < data.inputs.size() && data.inputs[desc.index]) {
                return get_buffer_arg(data.inputs[desc.index]);
            }
            break;
        case args_t::INPUT_OF_FUSED_PRIMITIVE:
            if (desc.index < data.fused_op_inputs.size() && data.fused_op_inputs[desc.index]) {
                return get_buffer_arg(data.fused_op_inputs[desc.index]);
            }
            break;
        case args_t::INTERNAL_BUFFER:
            if (desc.index < data.intermediates.size() && data.intermediates[desc.index]) {
                return get_buffer_arg(data.intermediates[desc.index]);
            }
            break;
        case args_t::OUTPUT:
            if (desc.index < data.outputs.size() && data.outputs[desc.index]) {
                return get_buffer_arg(data.outputs[desc.index]);
            }
            break;
        case args_t::WEIGHTS:
            return get_buffer_arg(data.weights);
            break;
        case args_t::BIAS:
            return get_buffer_arg(data.bias);
            break;
        case args_t::WEIGHTS_ZERO_POINTS:
            return get_buffer_arg(data.weights_zero_points);
            break;
        case args_t::ACTIVATIONS_ZERO_POINTS:
            return get_buffer_arg(data.activations_zero_points);
            break;
        case args_t::COMPENSATION:
            return get_buffer_arg(data.compensation);
            break;
        case args_t::SCALE_TABLE:
            return get_buffer_arg(data.scale_table);
            break;
        case args_t::SLOPE:
            return get_buffer_arg(data.slope);
            break;
        case args_t::SCALAR:
            if (data.scalars && desc.index < data.scalars->size()) {
                const auto& scalar = (*data.scalars)[desc.index];
                switch (scalar.t) {
                    case scalar_t::UINT8:
                        return get_scalar_arg<uint8_t>(scalar.v.u8);
                        break;
                    case scalar_t::UINT16:
                        return get_scalar_arg<uint16_t>(scalar.v.u16);
                        break;
                    case scalar_t::UINT32:
                        return get_scalar_arg<uint32_t>(scalar.v.u32);
                        break;
                    case scalar_t::UINT64:
                        return get_scalar_arg<uint64_t>(scalar.v.u64);
                        break;
                    case scalar_t::INT8:
                        return get_scalar_arg<int8_t>(scalar.v.s8);
                        break;
                    case scalar_t::INT16:
                        return get_scalar_arg<int16_t>(scalar.v.s16);
                        break;
                    case scalar_t::INT32:
                        return get_scalar_arg<int32_t>(scalar.v.s32);
                        break;
                    case scalar_t::INT64:
                        return get_scalar_arg<int64_t>(scalar.v.s64);
                        break;
                    case scalar_t::FLOAT32:
                        return get_scalar_arg<float>(scalar.v.f32);
                        break;
                    case scalar_t::FLOAT64:
                        return get_scalar_arg<double>(scalar.v.f64);
                        break;
                    default:
                        break;
                }
            }
            break;
        case args_t::CELL:
            return get_buffer_arg(data.cell);
            break;
        case args_t::SHAPE_INFO:
            return get_buffer_arg(data.shape_info);
            break;
        default:
            break;
    }

    OPENVINO_THROW("");
}

}  // namespace ze
}  // namespace cldnn
