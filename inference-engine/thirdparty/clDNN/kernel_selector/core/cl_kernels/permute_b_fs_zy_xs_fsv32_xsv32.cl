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

KERNEL (permute_b_fs_zy_xs_fsv32_xsv32)(
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
    __local VTYPE read_buf[READ_BUF_WIDTH * TILE_SIZE_H * LWS];
    __local VTYPE transpose_buf[TRANS_BUF_WIDTH * TILE_SIZE_H * LWS];

    int local_id = get_local_id(0) * get_local_size(2) * get_local_size(1)
                    + get_local_id(1) * get_local_size(1)
                    + get_local_id(2);

    int local_buf_offset = local_id * (TILE_SIZE_W/VECTORWIDTH) * TILE_SIZE_H;

    if ((f < F_REMAINDER_ITEM) && (x < X_REMAINDER_ITEM)) {
        // read partial data
        for(int lh = 0; lh < TILE_SIZE_H; ++lh) {
            for(int lw = 0; lw < READ_BUF_WIDTH; ++lw) {
#if INPUT0_DIMS == 6
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, w, z, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
#elif INPUT0_DIMS == 5
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, z, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
#elif INPUT0_DIMS == 4
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
#endif
                int target_idx =  lh * TILE_SIZE_W/VECTORWIDTH + lw;
                read_buf[local_buf_offset + target_idx] = AS_VTYPE(VLOAD(0, input + input_idx));
            }
        }
        // transpose
        for(int lh = 0; lh < TILE_SIZE_H; ++lh) {
            for(int lw = 0; lw < READ_BUF_WIDTH; ++lw) {

                unsigned int src = local_buf_offset + lh * TILE_SIZE_W/VECTORWIDTH + lw;

                unsigned int dst_h = lw * VECTORWIDTH;
                unsigned int dst_w = lh / VECTORWIDTH;
                unsigned int dst_element = lh % VECTORWIDTH;
                unsigned int dst_h_pitch = TILE_SIZE_H/VECTORWIDTH;
                unroll_for (int i = 0; i < VECTORWIDTH; ++i) {
                    unsigned int dst = local_buf_offset + (dst_h + i) * dst_h_pitch + dst_w;
                    INPUT0_TYPE input_var = read_buf[src][i];
#if HAS_FUSED_OPS
                    FUSED_OPS;
                    transpose_buf[dst][dst_element] = FUSED_OPS_RESULT;
#else
                    transpose_buf[dst][dst_element] = ACTIVATION(read_buf[src][i], ACTIVATION_PARAMS);
#endif
                }
            }
        }
        // write to ddr
        for(int lh = 0; lh < TILE_SIZE_W; ++lh) {
            for(int lw = 0; lw < TRANS_BUF_WIDTH; ++lw) {
                // b, f, z, x, y
#if INPUT0_DIMS == 6
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + w * OUTPUT_FEATURE_PITCH + z * OUTPUT_W_PITCH + y * OUTPUT_Z_PITCH +  (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw*VECTORWIDTH) * OUTPUT_X_PITCH;
#elif INPUT0_DIMS == 5
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + z * OUTPUT_FEATURE_PITCH + y * OUTPUT_Z_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw*VECTORWIDTH) * OUTPUT_X_PITCH;
#else
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + y * OUTPUT_FEATURE_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw*VECTORWIDTH) * OUTPUT_X_PITCH;
#endif
                vstore8(transpose_buf[local_buf_offset + lh * TILE_SIZE_H/VECTORWIDTH + lw], 0, output + output_idx);
            }
        }
    } else if (f == F_REMAINDER_ITEM && x < X_REMAINDER_ITEM) {
        // (f_remainder_size, TILE_SIZE_W)
        // read
        for (int lh = 0; lh < F_REMAINDER_SIZE; ++lh) {
            for(int lw = 0; lw < READ_BUF_WIDTH; ++lw) {
              #if INPUT0_DIMS == 6
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, w, z, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
              #elif INPUT0_DIMS == 5
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, z, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
              #elif INPUT0_DIMS == 4
                unsigned int input_idx = b * INPUT0_GET_INDEX(b, TILE_SIZE_H * f + lh, y, TILE_SIZE_W * x + lw * VECTORWIDTH);
              #endif
                int target_idx = lh * TILE_SIZE_W/VECTORWIDTH + lw;
                read_buf[local_buf_offset + target_idx] = AS_VTYPE(VLOAD(0, input + input_idx));
            }
        }
        // transpose
        for(int lh = 0; lh < F_REMAINDER_SIZE; ++lh) {
            for(int lw = 0; lw < READ_BUF_WIDTH; ++lw) {

                unsigned int src = local_buf_offset + lh * TILE_SIZE_W/VECTORWIDTH + lw;

                unsigned int dst_h = lw * VECTORWIDTH;
                unsigned int dst_w = lh / VECTORWIDTH;
                unsigned int dst_element = lh % VECTORWIDTH;
                unsigned int dst_h_pitch = (F_REMAINDER_SIZE + VECTORWIDTH - 1) / VECTORWIDTH;
                unroll_for (int i = 0; i < VECTORWIDTH; ++i) {
                    unsigned int dst = local_buf_offset + (dst_h + i) * dst_h_pitch + dst_w;
                    INPUT0_TYPE input_var = read_buf[src][i];
#if HAS_FUSED_OPS
                    FUSED_OPS;
                    transpose_buf[dst][dst_element] = FUSED_OPS_RESULT;
#else
                    transpose_buf[dst][dst_element] = ACTIVATION(read_buf[src][i], ACTIVATION_PARAMS);
#endif
                }
            }
        }
        // write to ddr
        for(int lh = 0; lh < TILE_SIZE_W; ++lh) {
            for(int lw = 0; lw < (F_REMAINDER_SIZE + VECTORWIDTH - 1)/VECTORWIDTH; ++lw) {
                // b, f, z, x, y
#if INPUT0_DIMS == 6
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + w * OUTPUT_FEATURE_PITCH + z * OUTPUT_W_PITCH + y * OUTPUT_Z_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw) * OUTPUT_X_PITCH;
#elif INPUT0_DIMS == 5
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + z * OUTPUT_FEATURE_PITCH + y * OUTPUT_Z_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw) * OUTPUT_X_PITCH;
#else
                unsigned int output_idx = b * OUTPUT_BATCH_PITCH + y * OUTPUT_FEATURE_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H * f + lw) * OUTPUT_X_PITCH;
#endif
                if (lw >= ((F_REMAINDER_SIZE + VECTORWIDTH - 1)/VECTORWIDTH - 1)) {
                    for( int i = 0; i < F_REMAINDER_SIZE % VECTORWIDTH; ++i) {
                        output[output_idx + i] = transpose_buf[local_buf_offset + lh * (F_REMAINDER_SIZE + VECTORWIDTH - 1)/VECTORWIDTH][i];
                    }
                } else {
                    // still vector
                    vstore8(transpose_buf[local_buf_offset + lh * (F_REMAINDER_SIZE + VECTORWIDTH - 1)/VECTORWIDTH + lw], 0, output + output_idx);
                }
            }
        }
    } else if (f < F_REMAINDER_ITEM && x == X_REMAINDER_ITEM) {
        // (TILE_SIZE_H, x_remainder_size)
    } else { // f == X_REMAINDER_ITEM && x == X_REMAINDER_ITEM)
        // point by point
    }
}
