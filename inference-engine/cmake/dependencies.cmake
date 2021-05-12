# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

# we have number of dependencies stored on ftp

if (CMAKE_CROSSCOMPILING)
    set(CMAKE_STAGING_PREFIX "${TEMP}")
endif()

include(ExternalProject)

if (ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()


if (ENABLE_DATA)
    add_models_repo(${ENABLE_DATA} "data:https://github.com/openvinotoolkit/testdata.git")
    set(MODELS_PATH "${TEMP}/models/src/data")
    set(DATA_PATH "${MODELS_PATH}")
endif()

message(STATUS "MODELS_PATH=" ${MODELS_PATH})

fetch_models_and_validation_set()

if(COMMAND get_linux_name)
    get_linux_name(LINUX_OS_NAME)
endif()

include(CMakeParseArguments)

if (ENABLE_MYRIAD)
    include(cmake/vpu_dependencies.cmake)
endif()

## enable cblas_gemm from OpenBLAS package
if (ENABLE_MKL_DNN AND GEMM STREQUAL "OPENBLAS")
    if(AARCH64)
        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(WARNING "OpenBLAS is not found!")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            reset_deps_cache(OpenBLAS_DIR)

            RESOLVE_DEPENDENCY(OpenBLAS
                    ARCHIVE_LIN "keembay/openblas_0.3.7_yocto_kmb.tar.xz"
                    TARGET_PATH "${TEMP}/openblas_0.3.7_yocto_kmb"
                    ENVIRONMENT "OpenBLAS_DIR"
                    SHA256 "c75aac901d5297d6d60a4b1f941f0335d8fd7f52e0dff8c445f644e2e45e6fba")

            update_deps_cache(OpenBLAS_DIR "${OpenBLAS}/lib/cmake/openblas" "Path to OpenBLAS package folder")

            find_package(OpenBLAS QUIET)

            if(OpenBLAS_FOUND)
                set(BLAS_FOUND TRUE)
                set(BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
                set(BLAS_LIBRARIES ${OpenBLAS_LIBRARIES})
            endif()

            unset(IE_PATH_TO_DEPS)
        endif()
    endif()

    if(NOT BLAS_LIBRARIES OR NOT BLAS_INCLUDE_DIRS)
        find_package(BLAS REQUIRED)

        if(BLAS_FOUND)
            find_path(BLAS_INCLUDE_DIRS cblas.h)
        else()
            message(ERROR "OpenBLAS not found: install OpenBLAS or set -DBLAS_INCLUDE_DIRS=<path to dir with cblas.h> and -DBLAS_LIBRARIES=<path to libopenblas.so or openblas.lib>")
        endif()
    endif()

    debug_message(STATUS "openblas=" ${BLAS_LIBRARIES})
endif ()

## MKL-ML package
if (GEMM STREQUAL "MKL")
    if(NOT MKLROOT)
        message(FATAL_ERROR "MKLROOT not found: install MKL and set -DMKLROOT=<path_to_MKL>")
    endif()
    set(MKL ${MKLROOT})
    debug_message(STATUS "mkl_ml=" ${MKLROOT})
endif ()

## Intel OMP package
if (THREADING STREQUAL "OMP")
    reset_deps_cache(OMP)
    if (WIN32 AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_WIN "iomp.zip"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "62c68646747fb10f19b53217cb04a1e10ff93606f992e6b35eb8c31187c68fbf")
    elseif(LINUX AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_LIN "iomp.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "7832b16d82513ee880d97c27c7626f9525ebd678decf6a8fe6c38550f73227d9")
    elseif(APPLE AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_MAC "iomp_20190130_mac.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "591ea4a7e08bbe0062648916f42bded71d24c27f00af30a8f31a29b5878ea0cc")
    else()
        message(FATAL_ERROR "Intel OMP is not available on current platform")
    endif()
    update_deps_cache(OMP "${OMP}" "Path to OMP root folder")
    log_rpath_from_dir(OMP "${OMP}/lib")
    debug_message(STATUS "intel_omp=" ${OMP})
    
    ie_cpack_add_component(omp REQUIRED)
    file(GLOB_RECURSE source_list "${OMP}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
    install(FILES ${source_list} 
            DESTINATION "deployment_tools/inference_engine/external/omp/lib"
            COMPONENT omp)
    
endif ()

## TBB package
if (THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO")
    reset_deps_cache(TBBROOT TBB_DIR)

    if (WIN32 AND X86_64)
        #TODO: add target_path to be platform specific as well, to avoid following if
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_WIN "tbb2020_20200415_win.zip"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "f1c9b9e2861efdaa01552bd25312ccbc5feeb45551e5f91ae61e29221c5c1479")
        RESOLVE_DEPENDENCY(TBBBIND_2_4
                ARCHIVE_WIN "tbbbind_2_4_static_win.zip"
                TARGET_PATH "${TEMP}/tbbbind_2_4"
                ENVIRONMENT "TBBBIND_2_4_ROOT"
                SHA256 "1a3a05082cc5ef1a764d635793be347b82c795f0e9ced771515fc3706a4dc4f0")
    elseif(ANDROID)  # Should be before LINUX due LINUX is detected as well
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_ANDROID "tbb2020_20200404_android.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "f42d084224cc2d643314bd483ad180b081774608844000f132859fca3e9bf0ce")
    elseif(LINUX AND X86_64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "tbb2020_20200415_lin_strip.tgz"
                TARGET_PATH "${TEMP}/tbb"
                SHA256 "95b2f3b0b70c7376a0c7de351a355c2c514b42c4966e77e3e34271a599501008")
        RESOLVE_DEPENDENCY(TBBBIND_2_4
                ARCHIVE_LIN "tbbbind_2_4_static_lin.tgz"
                TARGET_PATH "${TEMP}/tbbbind_2_4"
                SHA256 "888582a94f81821f9894cc089db36d5a6c2e0b6998cfa1fec0c027f28c597ada")
    elseif(LINUX AND AARCH64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "keembay/tbb2020_38404_kmb_lic.tgz"
                TARGET_PATH "${TEMP}/tbb_yocto"
                ENVIRONMENT "TBBROOT"
                SHA256 "321261ff2eda6d4568a473cb883262bce77a93dac599f7bd65d2918bdee4d75b")
    elseif(APPLE AND X86_64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_MAC "tbb2020_20200404_mac.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "ad9cf52e657660058aa6c6844914bc0fc66241fec89a392d8b79a7ff69c3c7f6")
    else()
        message(FATAL_ERROR "TBB is not available on current platform")
    endif()

    update_deps_cache(TBBROOT "${TBB}" "Path to TBB root folder")
    update_deps_cache(TBB_DIR "${TBB}/cmake" "Path to TBB cmake folder")

    update_deps_cache(TBBBIND_2_4_DIR "${TBBBIND_2_4}/cmake" "Path to TBBBIND_2_4 cmake folder")

    if (WIN32)
        log_rpath_from_dir(TBB "${TBB}/bin")
    else ()
        log_rpath_from_dir(TBB "${TBB}/lib")
    endif ()
    debug_message(STATUS "tbb=" ${TBB})
endif ()

if (ENABLE_OPENCV)
    reset_deps_cache(OpenCV_DIR)

    set(OPENCV_VERSION "4.5.2")
    set(OPENCV_BUILD "076")
    set(OPENCV_BUILD_YOCTO "772")

    if (AARCH64)
        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(WARNING "OpenCV is not found!")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            set(OPENCV_SUFFIX "yocto_kmb")
            set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")

            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "23c250796ad5fc9db810e1680ccdb32c45dc0e50cace5e0f02b30faf652fe343")

            unset(IE_PATH_TO_DEPS)
        endif()
    else()
        if (WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_WIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "a14f872e6b63b6ac12c7ff47fa49e578d14c14433b57f5d85ab5dd48a079938c")
        elseif(APPLE AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_MAC "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "3e162f96e86cba8836618134831d9cf76df0438778b3e27e261dedad9254c514")
        elseif(LINUX)
            if (AARCH64)
                set(OPENCV_SUFFIX "yocto_kmb")
                set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")
            elseif (ARM)
                set(OPENCV_SUFFIX "debian9arm")
                set(OPENCV_HASH "4274f8c40b17215f4049096b524e4a330519f3e76813c5a3639b69c48633d34e")
            elseif ((LINUX_OS_NAME STREQUAL "CentOS 7" OR
                     CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9") AND X86_64)
                set(OPENCV_SUFFIX "centos7")
                set(OPENCV_HASH "5fa76985c84fe7c64531682ef0b272510c51ac0d0565622514edf1c88b33404a")
            elseif (LINUX_OS_NAME MATCHES "CentOS 8" AND X86_64)
                set(OPENCV_SUFFIX "centos8")
                set(OPENCV_HASH "db087dfd412eedb8161636ec083ada85ff278109948d1d62a06b0f52e1f04202")
            elseif (LINUX_OS_NAME STREQUAL "Ubuntu 16.04" AND X86_64)
                set(OPENCV_SUFFIX "ubuntu16")
                set(OPENCV_HASH "cd46831b4d8d1c0891d8d22ff5b2670d0a465a8a8285243059659a50ceeae2c3")
            elseif (LINUX_OS_NAME STREQUAL "Ubuntu 18.04" AND X86_64)
                set(OPENCV_SUFFIX "ubuntu18")
                set(OPENCV_HASH "db087dfd412eedb8161636ec083ada85ff278109948d1d62a06b0f52e1f04202")
            elseif ((LINUX_OS_NAME STREQUAL "Ubuntu 20.04" OR LINUX_OS_NAME STREQUAL "LinuxMint 20.1") AND X86_64)
                set(OPENCV_SUFFIX "ubuntu20")
                set(OPENCV_HASH "2fe7bbc40e1186eb8d099822038cae2821abf617ac7a16fadf98f377c723e268")
            elseif(NOT DEFINED OpenCV_DIR AND NOT DEFINED ENV{OpenCV_DIR})
                message(FATAL_ERROR "OpenCV is not available on current platform (${LINUX_OS_NAME})")
            endif()
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 ${OPENCV_HASH})
        endif()

    endif()

    if(ANDROID)
        set(ocv_cmake_path "${OPENCV}/sdk/native/jni/")
    else()
        set(ocv_cmake_path "${OPENCV}/cmake")
    endif()

    update_deps_cache(OpenCV_DIR "${ocv_cmake_path}" "Path to OpenCV package folder")

    if(WIN32)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../bin")
    elseif(ANDROID)
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../../../lib")
    else()
        log_rpath_from_dir(OPENCV "${OpenCV_DIR}/../lib")
    endif()
    debug_message(STATUS "opencv=" ${OPENCV})
else()
    reset_deps_cache(OpenCV_DIR)
endif()

# TODO: remove global CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${IEDevScripts_DIR}")
include(cmake/ie_parallel.cmake)

if (ENABLE_GNA)
    reset_deps_cache(
            GNA
            GNA_PLATFORM_DIR
            GNA_KERNEL_LIB_NAME
            GNA_LIBS_LIST
            GNA_LIB_DIR
            libGNA_INCLUDE_DIRS
            libGNA_LIBRARIES_BASE_PATH)
    if (GNA_LIBRARY_VERSION STREQUAL "GNA1")
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA/gna_20181120.zip"
                TARGET_PATH "${TEMP}/gna"
                SHA256 "b631d6cc5f6cca4a89a3f5dfa383066f3282fee25d633d9085c605bdd8090210")
    else()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA1_1401")
            set(GNA_VERSION "01.00.00.1401")
            set(GNA_HASH "cc954e67525006bf8bd353a6682e38bf208f6d74e973e0fc292850e721f17452")
        endif()
        if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
            set(GNA_VERSION "02.00.00.1191.0")
            set(GNA_HASH "a61b4a9133549b0a9f0b46d069f72906ced28bcbbe7d5c361e687645f53a1c8b")
        endif()

        set(FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/include)
        if (WIN32)
            LIST(APPEND FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/win64)
        else()
            LIST(APPEND FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/linux)
        endif()
     
        RESOLVE_DEPENDENCY(GNA
                ARCHIVE_UNIFIED "GNA/GNA_${GNA_VERSION}.zip"
                TARGET_PATH "${TEMP}/gna_${GNA_VERSION}"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                FILES_TO_EXTRACT FILES_TO_EXTRACT_LIST
                SHA256 ${GNA_HASH})
    endif()
    update_deps_cache(GNA "${GNA}" "Path to GNA root folder")
    debug_message(STATUS "gna=" ${GNA})
endif()

if (ENABLE_SPEECH_DEMO)
    reset_deps_cache(SPEECH_LIBS_AND_DEMOS)
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    else()
        message(WARNING "Unable to locate Speech Demo")
    endif()
    if(DEFINED IE_PATH_TO_DEPS)
        if (WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_WIN "speech_demo_1.0.0.755_windows.zip"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.755"
                    SHA256 "58adef14b8a749f70fa83888614cee34b941956e6e958e445e3f48885b3c20a0")
            debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
        elseif (LINUX AND X86_64)
            if (LINUX_OS_NAME STREQUAL "CentOS 7" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9")
                RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_LIN "speech_demo_1.0.0.755_centos.tgz"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.755"
                    SHA256 "716201e377714ac50f3909c445d36d47a089de50a557d8ef65232de040671188")
                debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
            else()
                RESOLVE_DEPENDENCY(SPEECH_LIBS_AND_DEMOS
                    ARCHIVE_LIN "speech_demo_1.0.0.755_linux.tgz"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                    TARGET_PATH "${TEMP}/speech_demo_1.0.0.755"
                    SHA256 "7714b8776ec0183ed73eed6d3d965ee6d5c15d2dc49ee5ae118cc368c89c7a9d")
                debug_message(STATUS "speech_libs_and_demos=" ${SPEECH_LIBS_AND_DEMOS})
            endif()
        else()
            message(FATAL_ERROR "Speech Demo is not available on current platform")
        endif()
        unset(IE_PATH_TO_DEPS)
    endif()
    update_deps_cache(SPEECH_LIBS_AND_DEMOS "${SPEECH_LIBS_AND_DEMOS}" "Path to SPEECH_LIBS_AND_DEMOS root folder")
endif()
