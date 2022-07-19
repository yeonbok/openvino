// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#define INPUT_0 input0[GET_INDEX(INPUT, 0)]
#define INPUT_1 input1[GET_INDEX(INPUT, 1)]
#define INPUT_2 input2[GET_INDEX(INPUT, 2)]

KERNEL(select)(
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{

 uint d1  = (uint) get_global_id(0);
 uint d2  = (uint) get_global_id(1);
 uint d34 = (uint) get_global_id(2);

 uint d3  = d34 % OUTPUT_SIZES[2];
 uint d4  = d34 / OUTPUT_SIZES[2];

uint output_offset = OUTPUT_OFFSET +
                     d1*OUTPUT_PITCHES[0] +
                     d2*OUTPUT_PITCHES[1] +
                     d3*OUTPUT_PITCHES[2] +
                     d4*OUTPUT_PITCHES[3];
OUTPUT_TYPE in1 = input1[GET_INDEX(INPUT, 1)];
OUTPUT_TYPE in2 = input2[GET_INDEX(INPUT, 2)];
//const OUTPUT_TYPE res = select(INPUT_2, INPUT_1, MASK);
OUTPUT_TYPE res = select(in2, in1, MASK);
//#if ORIG_PRIM_NAME == 88888
//    printf("2_Select_4142 output[%d] = %f (in1: %f, in2: %f, mask: %d\n", output_offset, res, INPUT_1, INPUT_2, MASK);
//#endif
#if ORIG_PRIM_NAME == 77777
//    if (output_offset == 0) {
//        printf("1_Select_4142 output[%d] = %f (in1: %f, in2: %f, mask: %d\n", output_offset, res, INPUT_1, INPUT_2, MASK);
//    }
#endif

output[output_offset] = res;
//#if ORIG_PRIM_NAME == 77777
//printf("s\n");
//#endif
}
