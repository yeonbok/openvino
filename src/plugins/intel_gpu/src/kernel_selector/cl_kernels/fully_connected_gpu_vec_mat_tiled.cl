// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(fully_connected_gpu_vec_mat_tiled) (
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    // GWS : [NUM_TILE_N, 1, 1]
    // WI: [1, TILE_N_SIZE]
    uint gid = (uint)get_global_id(0);
    uint out_b = 1, out_y = 1, out_x = 1;
    uint out_f = gid * TILE_N_SIZE;
    // initialize output : calculate TILE_N outputs
    OUTPUT_TYPE outputs[TILE_N_SIZE];
    for (int i = 0; i < TILE_N_SIZE; ++i) {
        outputs[i] = OUTPUT_VAL_ZERO;
    }

    // read weight (TILE_N_SIZE * TILE_N_SIZE) => store transposed
    // TODO : check K_leftover
    // Also lets make the TILE_N_SIZE == TILE_K_SIZE so that we can transpose in place
    int num_k_tiles = FILTER_IFM_NUM / TILE_K_SIZE;
    FILTER_TYPE weights_tile[TILE_N_SIZE * TILE_K_SIZE];
    int weight_offset_n = 0;
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // load weights TILE_K * TILE_N
        for (int k_ti = 0; k_ti < TILE_K_SIZE; ++k_ti) {
            for (int n_vi = 0; n_vi < TILE_N_SIZE / VSIZE; ++n_vi) {
                int org_weight_idx = TILE_N_SIZE * gid + weight_offset_n + n_vi * VSIZE;
                FILTER_VTYPE weights_vdata = VLOAD(0, weights + org_weight_idx);
                // unroll
                unroll_for (int k_vi = 0; k_vi < VSIZE; ++k_vi) {
                    // store in a transposed order
                    weights_tile[(n_vi * VSIZE + k_vi) * TILE_K_SIZE + k_ti] = weights_vdata[k_vi];
                } 
            }
            weight_offset_n += FILTER_OFM_NUM;
        }
        // do mac
        for (int n = 0; n < TILE_N_SIZE; ++n) {
            for (int k_v = 0; k_v < (TILE_K_SIZE / VSIZE); ++k_v) {
                INPUT_VTYPE input_vdata = VLOAD(0, input + (k_tile * TILE_K_SIZE + k_v * VSIZE));
                unroll_for ( int vi = 0; vi < VSIZE; ++vi) {
                    outputs[n] += input_vdata[vi] * weights_tile[n * TILE_K_SIZE + k_v * VSIZE + vi];
                }
            }
        }
        // TODO : if k_t is the last k tile, do post porocessing (bias, activation, fusion)
    }
    for (int n_v = 0; n_v < TILE_N_SIZE / VSIZE; ++n_v) {
        OUT_VTYPE out_vdata;
        unroll_for (int n_vi = 0; n_vi < VSIZE; ++n_vi) {
            out_vdata[n_vi] = outputs[n_v * VSIZE + n_vi];
        }
        VSTORE(out_vdata, 0, output + out_f + n_v * VSIZE);
    }
}