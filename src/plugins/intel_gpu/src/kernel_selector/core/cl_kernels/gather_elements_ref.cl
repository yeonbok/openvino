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
#if INPUT1_DIMS==4
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
    const uint b=dim2/INPUT1_FEATURE_NUM;
    const uint f=dim2%INPUT1_FEATURE_NUM;
    const uint y=dim1;
    const uint x=dim0;
    int axis_val=indices[INPUT1_GET_INDEX(b,f,y,x)];
    const uint out_idx=OUTPUT_GET_INDEX(b,f,y,x);
    if(axis_val<0)
        axis_val+=AXIS_LEN0;
    #if AXIS==0
        const uint in0_idx=INPUT0_GET_INDEX(axis_val,f,y,x);
    #elif AXIS==1
        const uint in0_idx=INPUT0_GET_INDEX(b,axis_val,y,x);
    #elif AXIS==2
        const uint in0_idx=INPUT0_GET_INDEX(b,f,axis_val,x);
    #else
        const uint in0_idx=INPUT0_GET_INDEX(b,f,y,axis_val);
    #endif
#elif INPUT1_DIMS==5
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
    const uint b=dim2/INPUT1_FEATURE_NUM;
    const uint f=dim2%INPUT1_FEATURE_NUM;
    const uint z=dim1/INPUT1_SIZE_Y;
    const uint y=dim1%INPUT1_SIZE_Y;
    const uint x=dim0;
    int axis_val=indices[INPUT1_GET_INDEX(b,f,z,y,x)];
    if(axis_val<0)
        axis_val+=AXIS_LEN0;
    #if AXIS==0
        const uint in0_idx=INPUT0_GET_INDEX(axis_val,f,z,y,x);
    #elif AXIS==1
        const uint in0_idx=INPUT0_GET_INDEX(b,axis_val,z,y,x);
    #elif AXIS==2
        const uint in0_idx=INPUT0_GET_INDEX(b,f,axis_val,y,x);
    #elif AXIS==3
        const uint in0_idx=INPUT0_GET_INDEX(b,f,z,axis_val,x);
    #else
        const uint in0_idx=INPUT0_GET_INDEX(b,f,z,y,axis_val);
    #endif
    const uint out_idx=OUTPUT_GET_INDEX(b,f,z,y,x);
#elif INPUT1_DIMS==6
#else
#endif

INPUT0_TYPE val = data[in0_idx];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
#undef AXIS
