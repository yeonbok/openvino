// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "cldnn_remote_context.h"
#include "cldnn_itt.h"
#include "cldnn_engine.h"
#include "cldnn/runtime/device_query.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {
CLDNNRemoteAllocator CLDNNRemoteBlobImpl::m_allocator;

CLDNNRemoteBlobImpl::CLDNNRemoteBlobImpl(ClContext::Ptr context,
    cldnn::stream& stream,
    const cldnn::layout& layout,
    cldnn::shared_handle mem,
    cldnn::shared_surface surf,
    uint32_t plane,
    BlobType mem_type) :
    m_context(context), m_stream(stream), m_layout(layout), m_mem_type(mem_type), m_mem(mem), m_surf(surf), m_plane(plane),
    _handle(nullptr), _allocator(nullptr), m_memObject(nullptr), lockedHolder(nullptr) {
}

ParamMap CLDNNRemoteBlobImpl::getParams() const {
    assert(m_memObject != nullptr);
    auto params = m_memObject->get_internal_params();

    switch (m_mem_type) {
    case BT_BUF_INTERNAL:
    case BT_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
#ifdef _WIN32
    case BT_DX_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(DX_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface }
        };
#endif
    case BT_IMG_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_IMAGE2D) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BT_SURF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface },
            { GPU_PARAM_KEY(VA_PLANE),  params.plane }
        };
    default:
        IE_THROW() << "Unsupported shared object type " << m_mem_type;
    }
}

bool CLDNNRemoteBlobImpl::deallocate() noexcept {
    m_memObject.reset();
    return m_memObject == nullptr;
}

bool CLDNNRemoteBlobImpl::is_allocated() const noexcept {
    return m_memObject != nullptr;
}

bool CLDNNRemoteBlobImpl::is_locked() const noexcept {
    return lockedHolder != nullptr;
}

void CLDNNRemoteBlobImpl::allocate() noexcept {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNRemoteBlobImpl::Allocate");
    assert(m_memObject == nullptr);

    auto _impl = getContextImpl(m_context.lock());
    _impl->acquire_lock();
    std::shared_ptr<cldnn::engine> eng = _impl->GetEngine();

    switch (m_mem_type) {
    case BlobType::BT_BUF_INTERNAL: {
        m_memObject = eng->allocate_memory(m_layout);
        break;
    }
    case BlobType::BT_BUF_SHARED: {
        m_memObject = eng->share_buffer(m_layout, m_mem);
        break;
    }
#ifdef _WIN32
    case BlobType::BT_SURF_SHARED: {
        m_memObject = eng->share_surface(m_layout, m_mem, m_plane);
        break;
    }
    case BlobType::BT_DX_BUF_SHARED: {
        m_memObject = eng->share_dx_buffer(m_layout, m_mem);
        break;
    }
#else
    case BlobType::BT_SURF_SHARED: {
        m_memObject = eng->share_surface(m_layout, m_surf, m_plane);
        break;
    }
#endif
    case BlobType::BT_IMG_SHARED: {
        m_memObject = eng->share_image(m_layout, m_mem);
        break;
    }
    default:
        m_memObject.reset();
    }
    _impl->release_lock();
}

const std::shared_ptr<IAllocator>& CLDNNRemoteBlobImpl::getAllocator() const noexcept {
    if (!_allocator) {
        _allocator = std::shared_ptr<IAllocator>(&m_allocator, [] (IAllocator*) {});
    }
    return _allocator;
};

std::string CLDNNRemoteBlobImpl::getDeviceName() const noexcept {
    return getContextImpl(m_context.lock())->getDeviceName();
};

std::shared_ptr<RemoteContext> CLDNNRemoteBlobImpl::getContext() const noexcept {
    return m_context.lock();
}

void CLDNNRemoteBlobImpl::lock() const {
    lockedHolder = std::unique_ptr<cldnn::mem_lock<uint8_t>>(new cldnn::mem_lock<uint8_t>(m_memObject, m_stream));
    auto ptr = lockedHolder->data();
    _handle = reinterpret_cast<void*>(ptr);
    m_allocator.regLockedBlob(_handle, this);
}

void CLDNNRemoteBlobImpl::unlock() const {
    lockedHolder.reset();
}

LockedMemory<void> CLDNNRemoteBlobImpl::buffer() noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator*>(&m_allocator), _handle, 0);
}

LockedMemory<const void> CLDNNRemoteBlobImpl::cbuffer() const noexcept {
    lock();
    return LockedMemory<const void>(reinterpret_cast<IAllocator*>(&m_allocator), _handle, 0);
}

LockedMemory<void> CLDNNRemoteBlobImpl::rwmap()noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

LockedMemory<const void> CLDNNRemoteBlobImpl::rmap() const noexcept {
    lock();
    return LockedMemory<const void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

LockedMemory<void> CLDNNRemoteBlobImpl::wmap()noexcept {
    lock();
    return LockedMemory<void>(reinterpret_cast<IAllocator *>(&m_allocator), _handle, 0);
}

void CLDNNRemoteAllocator::regLockedBlob(void* handle, const CLDNNRemoteBlobImpl* blob) {
    acquire_lock();
    auto iter = m_lockedBlobs.find(handle);
    if (iter == m_lockedBlobs.end()) {
        m_lockedBlobs.emplace(handle, blob);
    }
    release_lock();
}

void CLDNNRemoteAllocator::unlock(void* handle) noexcept {
    acquire_lock();
    auto iter = m_lockedBlobs.find(handle);
    if (iter != m_lockedBlobs.end()) {
        iter->second->unlock();
        m_lockedBlobs.erase(iter);
    }
    release_lock();
}

CLDNNExecutionContextImpl::CLDNNExecutionContextImpl(const std::shared_ptr<IInferencePlugin> plugin,
    const ParamMap& params,
    const Config& config) :
    m_plugin(plugin),
    m_type(ContextType::OCL),
    m_config(config),
    m_external_queue(nullptr),
    m_va_display(nullptr) {
    lock.clear(std::memory_order_relaxed);
    gpu_handle_param _context_id = nullptr;
    gpu_handle_param _va_device = nullptr;
    int ctx_device_id = 0;
    int target_tile_id = -1;

    if (params.size()) {
        // parameter map is non-empty
        std::string contextTypeStr = _StrFromParams(params, GPU_PARAM_KEY(CONTEXT_TYPE));

        if (GPU_PARAM_VALUE(OCL) == contextTypeStr) {
            _context_id = _ObjFromParamSimple<gpu_handle_param>(params, GPU_PARAM_KEY(OCL_CONTEXT));

            if (params.find(GPU_PARAM_KEY(OCL_QUEUE)) != params.end())
                m_external_queue = _ObjFromParamSimple<gpu_handle_param>(params, GPU_PARAM_KEY(OCL_QUEUE));

            if (params.find(GPU_PARAM_KEY(OCL_CONTEXT_DEVICE_ID)) != params.end())
                ctx_device_id = _ObjFromParamSimple<int>(params, GPU_PARAM_KEY(OCL_CONTEXT_DEVICE_ID));
        } else if (GPU_PARAM_VALUE(VA_SHARED) == contextTypeStr) {
            m_va_display = _va_device = _ObjFromParamSimple<gpu_handle_param>(params, GPU_PARAM_KEY(VA_DEVICE));
            m_type = ContextType::DEV_SHARED;
        } else {
            IE_THROW() << "Invalid execution context type" << contextTypeStr;
        }
        auto tile_id_itr = params.find(GPU_PARAM_KEY(TILE_ID));
        if (tile_id_itr != params.end()) {
            target_tile_id = tile_id_itr->second.as<int>();
        }
    }

    // TODO: Parameterize this based on plugin config and compilation options
    auto engine_type = cldnn::engine_types::ocl;
    auto runtime_type = cldnn::runtime_types::ocl;
    // Use actual runtime and engine types
    cldnn::device_query device_query(engine_type, runtime_type, _context_id, _va_device, ctx_device_id, target_tile_id);
    auto device_map = device_query.get_available_devices();

    auto iter = device_map.find(m_config.device_id);
    auto& dev = iter != device_map.end() ? iter->second : device_map.begin()->second;

    bool enable_profiling = (m_config.useProfiling ||
                            (m_config.tuningConfig.mode == cldnn::tuning_mode::tuning_tune_and_cache) ||
                            (m_config.tuningConfig.mode == cldnn::tuning_mode::tuning_retune_and_cache));

    auto engine_params = clDNNEngine::GetEngineParams(m_config, dev, m_external_queue);
    m_engine = cldnn::engine::create(engine_params.engine_type,
                                     engine_params.runtime_type, dev,
                                     cldnn::engine_configuration(enable_profiling,
                                         engine_params.queue_type,
                                         m_config.sources_dumps_dir,
                                         m_config.queuePriority,
                                         m_config.queueThrottle,
                                         m_config.memory_pool_on,
                                         engine_params.use_unified_shared_memory,
                                         m_config.kernels_cache_dir,
                                         m_config.throughput_streams),
                                     engine_params.task_executor);
}

ParamMap CLDNNExecutionContextImpl::getParams() const {
    ParamMap ret = { { GPU_PARAM_KEY(OCL_CONTEXT), m_engine->get_user_context() } };

    switch (m_type) {
    case OCL:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(OCL);
        ret[GPU_PARAM_KEY(OCL_QUEUE)] = static_cast<gpu_handle_param>(m_external_queue);
        break;
    case DEV_SHARED:
        ret[GPU_PARAM_KEY(CONTEXT_TYPE)] = GPU_PARAM_VALUE(VA_SHARED);
        ret[GPU_PARAM_KEY(VA_DEVICE)] = m_va_display;
        break;
    default:
        IE_THROW() << "Unsupported shared context type " << m_type;
    }

    return ret;
}

std::string CLDNNExecutionContextImpl::getDeviceName() const noexcept {
    auto devName = m_plugin.lock()->GetName();

    auto engine_type = cldnn::engine_types::ocl;
    auto runtime_type = cldnn::runtime_types::ocl;
    // Use actual runtime and engine types
    cldnn::device_query device_query(engine_type, runtime_type);
    auto all_devices = device_query.get_available_devices();
    auto current_device = m_engine->get_device();

    for (auto& kv : all_devices) {
        if (current_device->is_same(kv.second))
            return devName + "." + kv.first;
    }

    if (!m_config.device_id.empty())
        devName += "." + m_config.device_id;
    return devName;
}

};  // namespace CLDNNPlugin
