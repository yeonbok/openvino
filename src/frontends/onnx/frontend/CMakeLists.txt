# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT BUILD_SHARED_LIBS)
    file(GLOB_RECURSE op_list "src/op/*.cpp")
    set(static_reg_file ${CMAKE_CURRENT_BINARY_DIR}/static_reg.hpp)
    file(WRITE ${static_reg_file} "// Copyright (C) 2018-2025 Intel Corporation\n// SPDX-License-Identifier: Apache-2.0\n// Auto generated file, DO NOT EDIT INLINE\n\n")
    file(APPEND ${static_reg_file} "#include \"core/operator_set.hpp\"\n\n")
    file(APPEND ${static_reg_file} "#define ONNX_DECL_OP(op) extern ov::OutputVector op(const Node&)\n\n")
    file(APPEND ${static_reg_file} "namespace ov {\nnamespace frontend {\nnamespace onnx {\n")
    foreach(src ${op_list})
        file(READ ${src} source_code)
        string(REGEX MATCHALL "ONNX_OP([^;]+);" matches "${source_code}")
        foreach(match ${matches})
            if(${match} MATCHES "([a-z0-9_]+)::([a-z0-9_]+)::([a-z0-9_]+)")
                list(APPEND declarations ${CMAKE_MATCH_0})
            endif()
            list(APPEND registrations ${match})
        endforeach()
    endforeach()
    list(APPEND declarations "com_microsoft::opset_1::register_multiple_translators")
    list(APPEND registrations "com_microsoft::opset_1::register_multiple_translators()")
    list(SORT declarations)
    set(domain "")
    set(opset "")
    set(op_name, "")
    foreach(decl ${declarations})
        string(REGEX MATCH "([a-z0-9_]+)::([a-z0-9_]+)::([a-z0-9_]+)" matches ${decl})
        if(NOT domain STREQUAL CMAKE_MATCH_1)
            if(NOT opset STREQUAL "")
                file(APPEND ${static_reg_file} "}  // namespace ${opset}\n")
            endif()
            if(NOT domain STREQUAL "")
                file(APPEND ${static_reg_file} "}  // namespace ${domain}\n")
            endif()
            set(domain ${CMAKE_MATCH_1})
            set(opset "")
            file(APPEND ${static_reg_file} "namespace ${domain} {\n")
        endif()
        if(NOT opset STREQUAL CMAKE_MATCH_2)
            if(NOT opset STREQUAL "")
                file(APPEND ${static_reg_file} "}  // namespace ${opset}\n")
            endif()
            set(opset ${CMAKE_MATCH_2})
            file(APPEND ${static_reg_file} "namespace ${opset} {\n")
        endif()
        if(NOT op_name STREQUAL CMAKE_MATCH_3)
            set(op_name ${CMAKE_MATCH_3})
            if(NOT op_name STREQUAL "register_multiple_translators")
                file(APPEND ${static_reg_file} "ONNX_DECL_OP(${CMAKE_MATCH_3});\n")
            else()
                file(APPEND ${static_reg_file} "extern bool ${CMAKE_MATCH_3}(void);\n")
            endif()
        endif()
    endforeach()
    if(NOT opset STREQUAL "")
        file(APPEND ${static_reg_file} "}  // namespace ${opset}\n")
    endif()
    if(NOT domain STREQUAL "")
        file(APPEND ${static_reg_file} "}  // namespace ${domain}\n")
    endif()
    file(APPEND ${static_reg_file} "\nvoid static_lib_registration(void) {\n")
    foreach(reg ${registrations})
        string(REPLACE "ONNX_OP(" "ONNX_OP_M(" reg ${reg})
        file(APPEND ${static_reg_file} "    ${reg};\n")
    endforeach()
    file(APPEND ${static_reg_file} "}\n")
    file(APPEND ${static_reg_file} "}  // namespace onnx\n}  // namespace frontend\n}  // namespace ov\n#undef ONNX_DECL_OP\n")
endif()

ov_add_frontend(NAME onnx
                LINKABLE_FRONTEND
                PROTOBUF_REQUIRED
                PROTOBUF_LITE
                SKIP_NCC_STYLE
                FILEDESCRIPTION "FrontEnd to load and convert ONNX file format"
                LINK_LIBRARIES openvino_onnx_common openvino::core::dev)

set(ONNX_OPSET_VERSION 21 CACHE INTERNAL "Supported version of ONNX operator set")
target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_OPSET_VERSION=${ONNX_OPSET_VERSION})

if(BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PRIVATE ONNX_BUILD_SHARED=1)
endif()

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    SOURCE_DIRECTORIES "${${TARGET_NAME}_INCLUDE_DIR}"
                    DEFINITIONS
                        $<TARGET_PROPERTY:onnx,INTERFACE_COMPILE_DEFINITIONS>)
