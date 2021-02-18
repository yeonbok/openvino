// Copyright (c) 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "include/include_all.cl"
#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)

KERNEL (permute_tile_8x8)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = get_global_id(0);
    const uint f = (uint)get_global_id(2) % NFEATURE_TILES;
    const uint b = (uint)get_global_id(2) / NFEATURE_TILES;

#if INPUT0_DIMS == 4 && OUTPUT_DIMS == 4
    //|dim2:bf|dim1:y|dim0:x
    const uint y = get_global_id(1);
#elif INPUT0_DIMS == 5 && OUTPUT_DIMS == 5
    //|dim2:bf|dim1:yz|dim0:x
    const uint z = get_global_id(1) / INPUT0_SIZE_Y;
    const uint y = get_global_id(1) % INPUT0_SIZE_Y;   
#elif INPUT0_DIMS == 6 && OUTPUT_DIMS == 6
    //|dim2:bf|dim1:wyz|dim0:x
    const uint y = get_global_id(1) % INPUT0_SIZE_Y;
    const uint z = get_global_id(1) / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint w = get_global_id(1) / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
#endif
    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];

    int local_id = get_local_id(0) * get_local_size(2) * get_local_size(1)
                    + get_local_id(1) * get_local_size(2)
                    + get_local_id(2);

    int local_buf_offset = local_id * LOCAL_BUF_STRIDE;

    int src_height = TILE_SIZE;
    int src_width = N_VECTORS_IN_TILE;
#ifdef F_REMAINDER_ITEM
    if (F_REMAINDER_CONDITION)
        src_height = F_REMAINDER_SIZE;
#endif

#ifdef X_REMAINDER_ITEM
        if (X_REMAINDER_CONDITION)
            src_width = X_REMAINDER_SIZE_AS_VECTOR;
#endif
    for (int lh = 0; lh < src_height; ++lh) {
        for (int lw = 0; lw < src_width; ++lw) {
            // read
            unsigned int input_idx = INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx + lw));
            // transpose
            unsigned int dst_h = lw * VECTORWIDTH;
            unsigned int dst_w = lh / VECTORWIDTH;
            unsigned int dst_element = lh % VECTORWIDTH;
            unsigned int dst_h_pitch = N_VECTORS_IN_TILE;
            int read_fragment_width = VECTORWIDTH;
#ifdef X_REMAINDER_ITEM
            if (X_REMAINDER_CONDITION) {
                read_fragment_width = (lw == (src_width - 1)) ? X_REMAINDER_SIZE % VECTORWIDTH : VECTORWIDTH;
            } 
#endif
            unroll_for (int i = 0; i < read_fragment_width; ++i) {
                unsigned int dst = local_buf_offset + (dst_h + i) * dst_h_pitch + dst_w;
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = read_data[i];
                FUSED_OPS;
                transpose_buf[dst][dst_element] = FUSED_OPS_RESULT;
#else
                transpose_buf[dst][dst_element] = ACTIVATION(read_data[i], ACTIVATION_PARAMS);
#endif
            }
        }
    }
    // write to ddr
    int dst_height = TILE_SIZE;
    int dst_width  = N_VECTORS_IN_TILE;
#ifdef X_REMAINDER_ITEM
    if (X_REMAINDER_CONDITION)
        dst_height = X_REMAINDER_SIZE;
#endif

#ifdef F_REMAINDER_ITEM
    if (F_REMAINDER_CONDITION)
        dst_width = F_REMAINDER_SIZE_AS_VECTOR;
    else
        dst_width = N_VECTORS_IN_TILE;
#endif

    for(int lh = 0; lh < dst_height; ++lh) {
        for(int lw = 0; lw < dst_width; ++lw) {
            // b, f, z, x, y
            unsigned int output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
#ifdef F_REMAINDER_ITEM
            if (F_REMAINDER_CONDITION && (lw == (CEIL_DIV(F_REMAINDER_SIZE, VECTORWIDTH) - 1))) {
                for ( int i = 0; i < F_REMAINDER_SIZE % VECTORWIDTH; ++i) {
                    output[output_idx + i] = transpose_buf[local_buf_offset + lh * CEIL_DIV(F_REMAINDER_SIZE, VECTORWIDTH) + lw][i];
                }
            } else {
                VSTORE(transpose_buf[local_buf_offset + lh * CEIL_DIV(F_REMAINDER_SIZE, VECTORWIDTH) + lw], 0, output + output_idx);
            }
#else
            VSTORE(transpose_buf[local_buf_offset + lh * N_VECTORS_IN_TILE + lw], 0, output + output_idx);
#endif
        }
    }
}
