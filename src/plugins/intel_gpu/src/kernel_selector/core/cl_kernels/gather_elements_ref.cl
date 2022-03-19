// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(gather_elements_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
#if AXIS==0
    #define AXIS_LEN0 INPUT0_BATCH_NUM
    #define AXIS_LEN1 INPUT1_BATCH_NUM
#elif AXIS==1
    #define AXIS_LEN0 INPUT0_FEATURE_NUM
    #define AXIS_LEN1 INPUT1_FEATURE_NUM
#elif AXIS==2
    #define AXIS_LEN0 INPUT0_SIZE_Y
    #define AXIS_LEN1 INPUT1_SIZE_Y
#else
    #define AXIS_LEN0 INPUT0_SIZE_X
    #define AXIS_LEN1 INPUT1_SIZE_X
#endif
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
    const uint b=dim0/INPUT1_FEATURE_NUM;
    const uint f=dim0%INPUT1_FEATURE_NUM;
    const uint y=dim1;
    const uint x=dim2;
    int axis_val=indices[INPUT1_GET_INDEX(b,f,y,x)];
    if(axis_val<0)
        axis_val+=AXIS_LEN0;
    #if AXIS==0
        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(axis_val,f,y,x)];
    #elif AXIS==1
        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,axis_val,y,x)];
    #elif AXIS==2
        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,f,axis_val,x)];
    #else
        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,f,y,axis_val)];
    #endif
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
