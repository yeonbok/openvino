/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"


__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global half *A, const global half *B, global float *C,
        const global int *A_offsets, const global int *B_offsets, const global int* C_offsets,
//        const global int *n_array, int k, local int *slm) {
        int m, const global int *n_array, int k) {
    printf("hello moe gemm micro\n");
    uint batch = get_group_id(2);
    A += A_offsets[batch];
    B += B_offsets[batch];
    C += C_offsets[batch];
    int n = n_array[0]; // TODO
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int gid2 = get_global_id(2);

    int lda = k;
    int ldb = k;

    uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);

    uint wg_i0 = get_group_id(0) * ugemm_moe_wg_tile_m;
    uint wg_j0 = get_group_id(1) * ugemm_moe_wg_tile_n;
    uint sg_i0 = wg_i0 + sg_i * ugemm_moe_sg_tile_m;
    uint sg_j0 = wg_j0 + sg_j * ugemm_moe_sg_tile_n;

    if (wg_j0 >= n) return;     /* early exit if outside batch */

    ugemm_moe_c_type c_tile = ugemm_moe(A, lda, B, ldb, m, n, k, wg_i0, wg_j0, 0, sg_i, sg_j);

    tile_store(c_tile, C, m, n, sg_i0, sg_j0);     // note oneDNN version needs a leading dimension parameter
}
