// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Wrappers from c++ function to c-style one
 * @file exception2status.hpp
 */
#pragma once

#include <string>

#include "description_buffer.hpp"

namespace InferenceEngine {

INFERENCE_ENGINE_API_CPP(StatusCode) ExceptionToStatus(const Exception& exception);

/**
 * @def TO_STATUS(x)
 * @brief Converts C++ exceptioned function call into a c-style one
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS(x)                                                                                            \
    try {                                                                                                       \
        x;                                                                                                      \
        return OK;                                                                                              \
    } catch (const ::InferenceEngine::Exception& iex) {                                                         \
        return InferenceEngine::DescriptionBuffer(InferenceEngine::ExceptionToStatus(iex), resp) << iex.what(); \
    } catch (const std::exception& ex) {                                                                        \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                            \
    } catch (...) {                                                                                             \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                  \
    }

/**
 * @def TO_STATUS_NO_RESP(x)
 * @brief Converts C++ exceptioned function call into a status code. Does not work with a ResponseDesc object
 * @ingroup ie_dev_api_error_debug
 */
#define TO_STATUS_NO_RESP(x)                                                                                        \
    try {                                                                                                           \
        x;                                                                                                          \
        return OK;                                                                                                  \
    } catch (const ::InferenceEngine::Exception& iex) {                                                             \
        return InferenceEngine::DescriptionBuffer(InferenceEngine::ExceptionToStatus(iex)) << iex.what();           \
    } catch (const std::exception& ex) {                                                                            \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR) << ex.what();                                      \
    } catch (...) {                                                                                                 \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                      \
    }

/**
 * @def NO_EXCEPT_CALL_RETURN_STATUS(x)
 * @brief Returns a status code of a called function, handles exeptions and converts to a status code.
 * @ingroup ie_dev_api_error_debug
 */
#define NO_EXCEPT_CALL_RETURN_STATUS(x)                                                                         \
    try {                                                                                                       \
        return x;                                                                                               \
    } catch (const ::InferenceEngine::Exception& iex) {                                                         \
        return InferenceEngine::DescriptionBuffer(InferenceEngine::ExceptionToStatus(iex), resp) << iex.what(); \
    } catch (const std::exception& ex) {                                                                        \
        return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();                            \
    } catch (...) {                                                                                             \
        return InferenceEngine::DescriptionBuffer(UNEXPECTED);                                                  \
    }

}  // namespace InferenceEngine
