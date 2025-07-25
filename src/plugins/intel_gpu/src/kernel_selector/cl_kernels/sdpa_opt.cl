// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]

inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#else
#if INPUT0_DIMS == 4
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 0 format
#endif
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT0_DIMS_ORDER
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
#else
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D(INPUT1, b, f, w, z, y, x);
#else
#if INPUT1_DIMS == 4
    return INPUT1_GET_INDEX(b, f, y, x);
#elif INPUT1_DIMS == 5
    return INPUT1_GET_INDEX(b, f, z, y, x);
#elif INPUT1_DIMS == 6
    return INPUT1_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT1_DIMS_ORDER
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
#else
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#if INPUT2_DIMS == 4
    return INPUT2_GET_INDEX(b, f, y, x);
#elif INPUT2_DIMS == 5
    return INPUT2_GET_INDEX(b, f, z, y, x);
#elif INPUT2_DIMS == 6
    return INPUT2_GET_INDEX(b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input2_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT2_DIMS_ORDER
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
#else
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

#ifdef BEAM_TABLE_TYPE
inline uint FUNC(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if BEAM_TABLE_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(BEAM_TABLE, b, f, w, z, y, x);
#else
#   error sdpa_opt.cl : Unsupported beam table format
#endif
}

inline uint FUNC(get_bt_index_key)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

inline uint FUNC(get_bt_index_value)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
}
#endif

#define OUTPUT_BLOCK_READ(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 1, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define VALUE_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT2_TYPE, 1, ptr, offset)
#define SUBGROUPS_PER_WG CEIL_DIV(V_HEAD_SIZE * SG_SCALE_FACTOR, SUBGROUP_SIZE)

#if IS_KV_COMPRESSED
#if COMPRESSED_PER_HEAD
    #define GET_COMPRESSION_INDEX(INPUT, b, f, y, x) GET_DATA_INDEX(INPUT, (b), (f), (y), (0));
#else
    #define GET_COMPRESSION_INDEX(INPUT, b, f, y, x) GET_DATA_INDEX(INPUT, (b), (0), (y), (0));
#endif
#endif

#ifdef SDPA_STAGE_0

#if HAS_SCALE_INPUT
#if HAS_ATTN_MASK_INPUT
#define SCALE_TYPE INPUT4_TYPE
#else
#define SCALE_TYPE INPUT3_TYPE
#endif
#endif

#if TARGET_SEQ_LEN_BLOCK_SIZE == 1
/* This version is used for 2nd token */

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, V_HEAD_SIZE * SG_SCALE_FACTOR)))
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global SCALE_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
#if IS_KV_COMPRESSED
    const __global KEY_COMPRESSION_SCALE_TYPE* key_scale,
    const __global VALUE_COMPRESSION_SCALE_TYPE* val_scale,
#endif
#ifdef BEAM_TABLE_TYPE
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
)
{
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / NUM_HEADS; /* BATCH dim */
    const uint b1_idx = batch_idx % NUM_HEADS; /* HEADS_NUM dim */
    const uint target_seq_idx = get_global_id(1);
    const uint lid = get_local_id(2);

#if SG_SCALE_FACTOR == 2
    const uint head_size_idx = lid % V_HEAD_SIZE;
#elif SG_SCALE_FACTOR == 1
    const uint head_size_idx = lid;
#else
    #error "sdpa_opt.cl: Unsupported scale factor"
#endif

#if SUBGROUPS_PER_WG > SUBGROUP_SIZE
    #error "sdpa_opt.cl: Number of subgroups per work group should be no more than subgroup_size"
#endif
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint partition_idx = get_group_id(2);
    const uint num_of_partitions = get_num_groups(2);
    const uint wi_num_per_partition = get_local_size(2);

    const uint start_partition_idx = partition_idx * SEQ_LEN_PARTITION_SIZE;
    const uint partition_seq_len =
        ((partition_idx + 1) < num_of_partitions) ? (SEQ_LEN_PARTITION_SIZE)
                                                  : (SOURCE_SEQ_LEN - partition_idx * SEQ_LEN_PARTITION_SIZE);

    // SLM for query inputs
    __local INPUT0_TYPE query_local[K_HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM for intermediate QK results
    __local SOFTMAX_ACCUMULATOR_TYPE qk_local[SEQ_LEN_PARTITION_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];
    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WG
    __local SOFTMAX_ACCUMULATOR_TYPE qk_max_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE qk_sum_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];

    {
        // Gemm1 and SoftMax calculation
        SOFTMAX_ACCUMULATOR_TYPE qk_max[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_MIN};
        for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
            qk_max[i] = SOFTMAX_ACCUMULATOR_VAL_MIN;
        }

        {
            // Gemm1 calculation
#if HAS_SCALE_INPUT
            const OUTPUT_TYPE scale_val = *scale;
#elif defined(STATIC_SCALE_VALUE)
            const OUTPUT_TYPE scale_val = TO_OUTPUT_TYPE(STATIC_SCALE_VALUE);
#else
            const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(V_HEAD_SIZE));
#endif
            #if K_HEAD_SIZE > V_HEAD_SIZE
            for (uint block_idx = sgid; block_idx < K_HEAD_SIZE / SUBGROUP_SIZE; block_idx++)
            #else
            const uint block_idx = sgid;
            #endif
            {
                // Query input loading to SLM
                #define QUERY_STEP_LOCAL SUBGROUP_SIZE * SUBGROUPS_PER_WG
                uint query_local_offset = block_idx * SUBGROUP_SIZE + sglid;
                const uint seq_idx_end = 1;
#ifdef INPUT0_DIMS_ORDER
                uint query_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx, (block_idx * SUBGROUP_SIZE));
                uint query_offset_next_seq = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx + 1, (block_idx * SUBGROUP_SIZE));
                const uint query_pitch = query_offset_next_seq - query_offset;
#else
                uint query_offset = INPUT0_GET_INDEX(b0_idx, b1_idx, target_seq_idx, (block_idx * SUBGROUP_SIZE));
                const uint query_pitch = QUERY_STEP_LOCAL;
#endif
#if SG_SCALE_FACTOR == 2
                if (block_idx < K_HEAD_SIZE / SUBGROUP_SIZE) {
#else
                {
#endif
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        #define QUERY_BLOCK_SIZE 1

                        INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, QUERY_BLOCK_SIZE, query_input, query_offset);
                        query_local[query_local_offset] = val * scale_val;
                        query_local_offset += QUERY_STEP_LOCAL;
                        query_offset += query_pitch;
                    }
                }
                #undef QUERY_BLOCK_SIZE
                #undef QUERY_STEP

            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Main Gemm1 calculation loop
            // Each SG performs element-wise multiplications of Q[HEAD_SIZE]xK[HEAD_SIZE] values
            // HEAD_SIZE / SUBGROUPS_PER_WG times in the loop and saves the result to the qk_local SLM buffer
            for (uint seq_len = sgid; seq_len < partition_seq_len; seq_len += SUBGROUPS_PER_WG) {
#ifdef BEAM_TABLE_TYPE
                const uint b_idx = beam_table[FUNC_CALL(get_bt_index_key)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len, 0)];
#else
                const uint b_idx = b0_idx;
#endif

#ifdef INPUT1_DIMS_ORDER
                uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, 0);
#else
                uint key_offset = INPUT1_GET_INDEX(b_idx, b1_idx, start_partition_idx + seq_len, 0);
#endif

                SOFTMAX_ACCUMULATOR_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};

#if IS_KV_COMPRESSED
                const uint comp_offset = GET_COMPRESSION_INDEX(KEY_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + seq_len, 0);
                KEY_COMPRESSION_SCALE_TYPE comp_scale = key_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                KEY_COMPRESSION_SCALE_TYPE comp_zp = key_scale[comp_offset + 1];
#endif
#endif
                uint head_idx_index = 0;
                #define KEY_BLOCK_SIZE 8
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define KEY_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(KEY_COMPRESSION_SCALE_TYPE, KEY_BLOCK_SIZE)
                    #define TO_KEY_BLOCK_UNCOMPRESSED_TYPE(val) CAT(convert_, KEY_BLOCK_UNCOMPRESSED)(val)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vec_packed = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed) - comp_zp) * comp_scale;
#elif IS_KV_COMPRESSED
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed)) * comp_scale;
#else
                    KEY_BLOCK key_vals = key_vec_packed;
#endif

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(query_vals_reg[i]), TO_SOFTMAX_ACCUMULATOR_TYPE(key_vals[i]), acc[seq_idx]);
                        }

                        query_offset += K_HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 4
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define KEY_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(KEY_COMPRESSION_SCALE_TYPE, KEY_BLOCK_SIZE)
                    #define TO_KEY_BLOCK_UNCOMPRESSED_TYPE(val) CAT(convert_, KEY_BLOCK_UNCOMPRESSED)(val)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vec_packed = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed) - comp_zp) * comp_scale;
#elif IS_KV_COMPRESSED
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed)) * comp_scale;
#else
                    KEY_BLOCK key_vals = key_vec_packed;
#endif

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(query_vals_reg[i]), TO_SOFTMAX_ACCUMULATOR_TYPE(key_vals[i]), acc[seq_idx]);
                        }

                        query_offset += K_HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 2
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define KEY_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(KEY_COMPRESSION_SCALE_TYPE, KEY_BLOCK_SIZE)
                    #define TO_KEY_BLOCK_UNCOMPRESSED_TYPE(val) CAT(convert_, KEY_BLOCK_UNCOMPRESSED)(val)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vec_packed = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed) - comp_zp) * comp_scale;
#elif IS_KV_COMPRESSED
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed)) * comp_scale;
#else
                    KEY_BLOCK key_vals = key_vec_packed;
#endif

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg[i] = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            acc[seq_idx] = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(query_vals_reg[i]), TO_SOFTMAX_ACCUMULATOR_TYPE(key_vals[i]), acc[seq_idx]);
                        }

                        query_offset += K_HEAD_SIZE;
                    }
                }

                #define KEY_BLOCK_SIZE 1
                for (; head_idx_index + (KEY_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * KEY_BLOCK_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, KEY_BLOCK_SIZE, ptr, offset);
                    #define KEY_BLOCK MAKE_VECTOR_TYPE(INPUT1_TYPE, KEY_BLOCK_SIZE)
                    #define KEY_BLOCK_UNCOMPRESSED MAKE_VECTOR_TYPE(KEY_COMPRESSION_SCALE_TYPE, KEY_BLOCK_SIZE)
                    #define TO_KEY_BLOCK_UNCOMPRESSED_TYPE(val) CAT(convert_, KEY_BLOCK_UNCOMPRESSED)(val)
                    #define QUERY_BLOCK MAKE_VECTOR_TYPE(INPUT0_TYPE, KEY_BLOCK_SIZE)

                    KEY_BLOCK key_vec_packed = KEY_BLOCK_READ(key_input, key_offset + head_idx_index);
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed) - comp_zp) * comp_scale;
#elif IS_KV_COMPRESSED
                    KEY_BLOCK_UNCOMPRESSED key_vals = (TO_KEY_BLOCK_UNCOMPRESSED_TYPE(key_vec_packed)) * comp_scale;
#else
                    KEY_BLOCK key_vals = key_vec_packed;
#endif

                    uint query_offset = head_idx_index + sglid;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        QUERY_BLOCK query_vals_reg;
                        unroll_for(uint i = 0; i < KEY_BLOCK_SIZE; i++) {
                            query_vals_reg = query_local[query_offset + i * SUBGROUP_SIZE];
                        }

                        acc[seq_idx] = mad(TO_SOFTMAX_ACCUMULATOR_TYPE(query_vals_reg), TO_SOFTMAX_ACCUMULATOR_TYPE(key_vals), acc[seq_idx]);
                        query_offset += K_HEAD_SIZE;
                    }
                }

                // Sum up all accumulators accross single SG and save result to SLM
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = sub_group_reduce_add(acc[seq_idx]);
                    qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = acc[seq_idx];
                }
            }

            {
                // Wait until all SG finishes their calculations and apply scale and attention mask to the results
                barrier(CLK_LOCAL_MEM_FENCE);

                SOFTMAX_ACCUMULATOR_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
                const uint seq_idx_end = 1;
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    // Iterate over all values QK values in SLM and apply scale and attention mask
                    for (uint seq_len = sgid * SUBGROUP_SIZE + sglid; seq_len < partition_seq_len; seq_len += (SUBGROUPS_PER_WG * SUBGROUP_SIZE)) {
                        // Read value from SLM and apply scale
                        qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];

                        // Apply attention mask
#if IS_CAUSAL
                        if (start_partition_idx + seq_len > target_seq_idx + seq_idx)
                            qk_val[seq_idx] += INPUT0_VAL_MIN;
#elif !IS_CAUSAL && HAS_ATTN_MASK_INPUT
                        const uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0_idx, b1_idx, target_seq_idx + seq_idx, start_partition_idx + seq_len);
                        qk_val[seq_idx] += attn_mask[attn_mask_offset];
#elif defined(STATIC_SCALAR_ATTN_MASK_VALUE)
                        qk_val[seq_idx] += STATIC_SCALAR_ATTN_MASK_VALUE;
#endif

                        // Update qk_max value
                        qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max[seq_idx], TO_SOFTMAX_ACCUMULATOR_TYPE(qk_val[seq_idx]));

                        // Save modified qk value back to SLM
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len] = qk_val[seq_idx];
                    }
                }
            }
        } // Gemm1 calculation end

        {
            // SoftMax calculation
            const uint seq_idx_end = 1;
            // Find the maximum value of qk in the subgroup
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            // Find the maximum value of qk across all subgroups in the workgroup
            if (sglid == 0) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = qk_max[seq_idx];
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_max[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_MIN;

                if (sglid < SUBGROUPS_PER_WG)
                    qk_max[seq_idx] = qk_max_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Final maximum value of qk after reduction across all subgroups
                qk_max[seq_idx] = sub_group_reduce_max(qk_max[seq_idx]);
            }

            SOFTMAX_ACCUMULATOR_TYPE exp_sum[TARGET_SEQ_LEN_BLOCK_SIZE] = {SOFTMAX_ACCUMULATOR_VAL_ZERO};
            const uint qk_num_per_wi = CEIL_DIV(partition_seq_len, SUBGROUPS_PER_WG * SUBGROUP_SIZE);
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + lid;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) - qk_max[seq_idx]);
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);

                        exp_sum[seq_idx] += qk_new;
                    }
                }
            }

            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);

                if (sglid == 0)
                    qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sgid] = exp_sum[seq_idx];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                exp_sum[seq_idx] = SOFTMAX_ACCUMULATOR_VAL_ZERO;

                if (sglid < SUBGROUPS_PER_WG)
                    exp_sum[seq_idx] = qk_sum_vals[seq_idx * SUBGROUPS_PER_WG + sglid];

                // Find the final sum of all exp_sum[seq_idx] values in workgroup
                exp_sum[seq_idx] = sub_group_reduce_add(exp_sum[seq_idx]);
            }

            // const SOFTMAX_ACCUMULATOR_TYPE inv_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ONE / exp_sum[seq_idx];
            for (uint qk_idx = 0; qk_idx < qk_num_per_wi; qk_idx++) {
                const uint local_data_idx = qk_idx * (SUBGROUPS_PER_WG * SUBGROUP_SIZE) + lid;
                if (local_data_idx < partition_seq_len) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        SOFTMAX_ACCUMULATOR_TYPE qk_new = TO_SOFTMAX_ACCUMULATOR_TYPE(qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx]) / exp_sum[seq_idx];
                        qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + local_data_idx] = TO_OUTPUT_TYPE(qk_new);
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            {
                // If the number of partitions is greater than 1, save exm_sums and max_logits to the temporary buffers
                // Use single WI in the WG, since all the WIs have the same value
                if (num_of_partitions > 1 && lid == 0) {
                    for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        const uint exp_sums_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                                     b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                                     (seq_idx + target_seq_idx) * (num_of_partitions) +
                                                     partition_idx;
                        exp_sums[exp_sums_offset] = exp_sum[seq_idx];

                        const uint max_logits_offset = exp_sums_offset;
                        max_logits[max_logits_offset] = qk_max[seq_idx];
                    }
                }
            }
        } // SoftMax calculation end
    } // Gemm1 + SoftMax calculations end

    {
        // Gemm2 calculation
        OUTPUT_TYPE acc[TARGET_SEQ_LEN_BLOCK_SIZE] = {OUTPUT_VAL_ZERO};
#ifndef BEAM_TABLE_TYPE
#ifdef INPUT2_DIMS_ORDER
        uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 0, 0);
        uint value_offset_next_seq = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 1, 0);
        const uint value_pitch = value_offset_next_seq - value_offset;
#else
        const uint value_pitch = V_HEAD_SIZE;
#endif
#endif

#if SG_SCALE_FACTOR > 1
        const uint seq_len_start = (sgid / (V_HEAD_SIZE / SUBGROUP_SIZE)) * (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR / SUBGROUP_SIZE);
        const uint seq_len_end = min(seq_len_start + (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR / SUBGROUP_SIZE), partition_seq_len / SUBGROUP_SIZE);
#else
        const uint seq_len_start = 0;
        const uint seq_len_end = partition_seq_len / SUBGROUP_SIZE;
#endif

        for (uint seq_len = seq_len_start; seq_len < seq_len_end; seq_len++) {
#ifdef BEAM_TABLE_TYPE
            const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE)];
            uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE);
#else
            const uint b_idx = b0_idx;
#ifdef INPUT2_DIMS_ORDER
            uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
#else
            uint value_offset = INPUT2_GET_INDEX(b_idx, b1_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
#endif
#endif

#if IS_KV_COMPRESSED
            const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, 0);
            VALUE_COMPRESSION_SCALE_TYPE comp_scale = val_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
            VALUE_COMPRESSION_SCALE_TYPE comp_zp = val_scale[comp_offset + 1];
#endif
#endif

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len * SUBGROUP_SIZE + sglid];
            }

            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
#ifdef BEAM_TABLE_TYPE
                const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, i));
#else
                const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, value_offset);
#endif

#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
#elif IS_KV_COMPRESSED
                VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, i));
#else
                INPUT2_TYPE value_val = value_packed;
#endif
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    acc[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc[seq_idx]);
                }

#ifndef BEAM_TABLE_TYPE
                value_offset += value_pitch;
#endif
            }
        }


#if SG_SCALE_FACTOR > 1
        if (sgid >= V_HEAD_SIZE / SUBGROUP_SIZE) {
#endif

        for (uint seq_len = (partition_seq_len / SUBGROUP_SIZE) * SUBGROUP_SIZE; seq_len < partition_seq_len; seq_len++) {
#ifdef BEAM_TABLE_TYPE
            const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len, head_size_idx)];
#else
            const uint b_idx = b0_idx;
#endif

#ifdef INPUT2_DIMS_ORDER
            const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + seq_len, head_size_idx);
#else
            const uint value_offset = INPUT2_GET_INDEX(b_idx, b1_idx, start_partition_idx + seq_len, head_size_idx);
#endif

#if IS_KV_COMPRESSED
            const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + seq_len, 0);
            VALUE_COMPRESSION_SCALE_TYPE comp_scale = val_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
            VALUE_COMPRESSION_SCALE_TYPE comp_zp = val_scale[comp_offset + 1];
#endif
#endif

            OUTPUT_TYPE qk_val[TARGET_SEQ_LEN_BLOCK_SIZE];
            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                qk_val[seq_idx] = qk_local[seq_idx * SEQ_LEN_PARTITION_SIZE + seq_len];
            }

            const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, value_offset);
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
            const VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - comp_zp) * comp_scale;
#elif IS_KV_COMPRESSED
            const VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * comp_scale);
#else
            const INPUT2_TYPE value_val = value_packed;
#endif

            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                acc[seq_idx] = mad(qk_val[seq_idx], value_val, acc[seq_idx]);
            }
        }

#if SG_SCALE_FACTOR > 1
        } // if (sgid >= V_HEAD_SIZE / SUBGROUP_SIZE)
#endif

#if SG_SCALE_FACTOR > 1
        if ((partition_seq_len > (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR)) || (partition_seq_len % SUBGROUP_SIZE != 0)) {
            if (sgid >= V_HEAD_SIZE / SUBGROUP_SIZE) {
                // Reuse query_local SLM to sum-up results between two groups of subgroups
                query_local[head_size_idx] = acc[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (sgid < V_HEAD_SIZE / SUBGROUP_SIZE) {
                acc[0] += query_local[head_size_idx];
            }
        }
#endif

        // If the number of partitions is greater than 1, save results to the temporary buffer;
        // otherwise, save results directly to the main output.
#if SG_SCALE_FACTOR > 1
        if (sgid < V_HEAD_SIZE / SUBGROUP_SIZE) {
#endif
        if (num_of_partitions > 1) {
            const uint seq_idx_end = 1;
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                // Data layout of tmp_output buf: [batch, heads_num, q_len, partition_idx, head_size]
                const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * V_HEAD_SIZE) +
                                            b1_idx * (TARGET_SEQ_LEN * num_of_partitions * V_HEAD_SIZE) +
                                            (target_seq_idx + seq_idx) * (num_of_partitions * V_HEAD_SIZE) +
                                            partition_idx * (V_HEAD_SIZE) +
                                            head_size_idx;
                tmp_out[tmp_out_offset] = acc[seq_idx];
            }
        } else {
            const uint seq_idx_end = 1;
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                const uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx + seq_idx, head_size_idx);
                output[output_offset] = acc[seq_idx];
            }
        }
#if SG_SCALE_FACTOR > 1
        } // if (sgid < V_HEAD_SIZE / SUBGROUP_SIZE) {
#endif
    } // Gemm2 calculation end
}

#else
/* This version is used for 1st token */

#if IS_PAGED_ATTENTION
    #define SOURCE_SEQ_LEN (subsequence_begins[gws_seq_indexes_correspondence[((uint)get_global_id(1))] + 1] - subsequence_begins[gws_seq_indexes_correspondence[((uint)get_global_id(1))]])

    #define TARGET_SEQ_LEN (subsequence_begins[gws_seq_indexes_correspondence[((uint)get_global_id(1))] + 1] - subsequence_begins[gws_seq_indexes_correspondence[((uint)get_global_id(1))]])

    #define PA_BUFFERS      , subsequence_begins,                                 \
                              blocked_indexes_start,                              \
                              blocked_indexes_end,                                \
                              gws_seq_indexes_correspondence

    #define PA_BUFFERS_ARGS , const __global INPUT3_TYPE* subsequence_begins,     \
                              const __global int* blocked_indexes_start,          \
                              const __global int* blocked_indexes_end,            \
                              const __global int* gws_seq_indexes_correspondence
#else
    #define PA_BUFFERS
    #define PA_BUFFERS_ARGS
#endif
#if HAS_ATTN_MASK_INPUT
    #define ATTN_MASK_BUFFER , attn_mask
    #define ATTN_MASK_BUFFER_ARG , const __global INPUT3_TYPE* attn_mask
#else
    #define ATTN_MASK_BUFFER
    #define ATTN_MASK_BUFFER_ARG
#endif

#if HAS_SCALE_INPUT
    #define ATTN_SCALE_BUFFER , scale
    #define ATTN_SCALE_BUFFER_ARG , const __global SCALE_TYPE* scale
#else
    #define ATTN_SCALE_BUFFER
    #define ATTN_SCALE_BUFFER_ARG
#endif

// Applying scales to query input improves the accuracy, but leads to performance drop for FP16 KV-cache case,
// so use it only for compressed version
#if IS_KV_COMPRESSED
#define APPLY_SCALES_TO_QUERY 1
#endif

#define MASK_VECTOR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)

inline MASK_VECTOR_TYPE FUNC(load_attn_mask)(OPTIONAL_SHAPE_INFO_ARG
                                             uint b0_idx,
                                             uint b1_idx,
                                             uint target_seq_idx,
                                             uint source_seq_idx
                                             ATTN_MASK_BUFFER_ARG
                                             ATTN_SCALE_BUFFER_ARG
                                             PA_BUFFERS_ARGS
                                             ) {
#ifdef STATIC_SCALAR_ATTN_MASK_VALUE
    MASK_VECTOR_TYPE mask_vec = STATIC_SCALAR_ATTN_MASK_VALUE;
#else
    MASK_VECTOR_TYPE mask_vec = INPUT0_VAL_ZERO;
#endif
#if !IS_CAUSAL && HAS_ATTN_MASK_INPUT
    const uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0_idx, b1_idx, target_seq_idx, source_seq_idx);
    if (target_seq_idx >= (uint)TARGET_SEQ_LEN) {
        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            mask_vec[i] = NAN;
        }
    } else {
        if (source_seq_idx + SUBGROUP_SIZE <= (uint)SOURCE_SEQ_LEN) {
            unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                const INPUT3_TYPE mask_val = attn_mask[attn_mask_offset + i];
                mask_vec[i] = mask_val;
            }
        } else {
            const uint max_mask_offset = min(source_seq_idx + SUBGROUP_SIZE, (uint)SOURCE_SEQ_LEN);
            for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                const INPUT3_TYPE mask_val = source_seq_idx + i < max_mask_offset ? attn_mask[attn_mask_offset + i] : NAN;
                mask_vec[i] = mask_val;
            }
        }
    }
#endif

#if !IS_CAUSAL && !HAS_ATTN_MASK_INPUT
    if (target_seq_idx >= (uint)TARGET_SEQ_LEN) {
        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            mask_vec[i] = NAN;
        }
    } else {
        const uint max_mask_offset = min(source_seq_idx + SUBGROUP_SIZE, (uint)SOURCE_SEQ_LEN);
        for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            mask_vec[i] = source_seq_idx + i < max_mask_offset ? 0 : NAN;
        }
    }
#endif

#if IS_CAUSAL
    if (target_seq_idx >= (uint)TARGET_SEQ_LEN) {
        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
            mask_vec[i] = NAN;
        }
    } else {
        for (uint i = 0; i < SUBGROUP_SIZE; i++) {
#if defined(IS_PAGED_ATTENTION) && SLIDING_WINDOW_SIZE != 0
            if ((source_seq_idx + i > target_seq_idx) ||
                (target_seq_idx >= SLIDING_WINDOW_SIZE && source_seq_idx + i < target_seq_idx - SLIDING_WINDOW_SIZE))
#else
            if (source_seq_idx + i > target_seq_idx)
#endif
                mask_vec[i] = NAN;
        }
    }
#endif

#if HAS_SCALE_INPUT
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / *scale;
#else
    const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(STATIC_SCALE_VALUE_INV);
#endif

    // Apply scale to attn_mask
#if IS_CAUSAL || HAS_ATTN_MASK_INPUT || defined(STATIC_SCALAR_ATTN_MASK_VALUE)
    mask_vec *= scale_val;
#endif

    return mask_vec;
}

#if IS_PAGED_ATTENTION && HAS_ALIBI
#if HAS_SCALE_INPUT && HAS_ATTN_MASK_INPUT
#define ALIBI_TYPE INPUT5_TYPE
#else
#define ALIBI_TYPE INPUT4_TYPE
#endif
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if IS_PAGED_ATTENTION
    const __global INPUT3_TYPE* subsequence_begins,
#endif
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global SCALE_TYPE* scale,
#endif
#if IS_PAGED_ATTENTION && HAS_ALIBI
    const __global ALIBI_TYPE* alibi_slopes,
#endif
    __global OUTPUT_TYPE* output,
#if IS_KV_COMPRESSED
    const __global KEY_COMPRESSION_SCALE_TYPE* key_scale,
    const __global VALUE_COMPRESSION_SCALE_TYPE* val_scale,
#endif
#ifdef BEAM_TABLE_TYPE
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
#if IS_PAGED_ATTENTION
    const __global int* blocked_indexes_start,
    const __global int* blocked_indexes_end,
    const __global int* gws_seq_indexes_correspondence
#if PAGED_ATTENTION_SCORES_OUTPUT
    , __global SOFTMAX_ACCUMULATOR_TYPE* softmax_results
    , const __global int* subsequence_offsets
#if IS_PAGED_ATTENTION && HAS_SCORE_AGGREGATION
    , const __global int* cumulative_score_aggregation_sum
#endif
    , __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums
    , __global SOFTMAX_ACCUMULATOR_TYPE* max_logits
    , __global OUTPUT_TYPE* tmp_out
    , const uint aligned_max_context_len
#endif
#else
    __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    __global OUTPUT_TYPE* tmp_out
#endif
)
{
#if TARGET_SEQ_LEN_BLOCK_SIZE != 16
    #error sdpa_opt.cl: unsupported TARGET_SEQ_LEN_BLOCK_SIZE
#endif

    // Define indexes variables using macro declarations to avoid register spills
    #define batch_idx ((uint)get_global_id(0))
    #define num_heads_dim ((uint)get_global_id(0))
    #define b0_idx (batch_idx / NUM_HEADS)
    #define b1_idx (batch_idx % NUM_HEADS)
    #define target_seq_dim ((uint)get_global_id(1))
#if IS_PAGED_ATTENTION
    #define target_seq_idx ((uint)block_start_pos - subsequence_begins[gws_seq_indexes_correspondence[target_seq_dim]])
#else
    #define target_seq_idx ((uint)get_global_id(1) * TARGET_SEQ_LEN_BLOCK_SIZE)
#endif
#if SG_SCALE_FACTOR > 1
    #define head_size_idx ((uint)get_local_id(2) % V_HEAD_SIZE)
#elif SG_SCALE_FACTOR == 1
    #define head_size_idx ((uint)get_local_id(2))
#else
    #error "sdpa_opt.cl: Unsupported scale factor"
#endif
    #define sglid (uint)get_sub_group_local_id()
    #define sgid (uint)get_sub_group_id()

    // SLM buffer for query inputs
    __local INPUT0_TYPE slm_query[K_HEAD_SIZE * TARGET_SEQ_LEN_BLOCK_SIZE];

    // SLM buffer for intermediate QK results
    __local OUTPUT_TYPE slm_qk_vals[TARGET_SEQ_LEN_BLOCK_SIZE][SEQ_LEN_PARTITION_SIZE];

    // SLM buffers for SoftMax calculation and qk_max/qk_sums results aggregation across all WGs
#if IS_FLASHATTEN_V2
    __local SOFTMAX_ACCUMULATOR_TYPE slm_qk_max_vals[TARGET_SEQ_LEN_BLOCK_SIZE][SUBGROUPS_PER_WG];
#else
    __local SOFTMAX_ACCUMULATOR_TYPE slm_qk_max_vals[SUBGROUPS_PER_WG][TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_vals[SUBGROUPS_PER_WG * TARGET_SEQ_LEN_BLOCK_SIZE];
#endif

    // SLM buffers for SoftMax recalculation for current iteration based on the previous results
    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_prev[TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max_val_prev[TARGET_SEQ_LEN_BLOCK_SIZE];
#if IS_FLASHATTEN_V2
    __local SOFTMAX_ACCUMULATOR_TYPE slm_update_factor[TARGET_SEQ_LEN_BLOCK_SIZE];
#else
    __local SOFTMAX_ACCUMULATOR_TYPE slm_exp_sum_cur[TARGET_SEQ_LEN_BLOCK_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE slm_max_val_cur[TARGET_SEQ_LEN_BLOCK_SIZE];
#endif

#if IS_PAGED_ATTENTION
    const uint block_start_pos = blocked_indexes_start[target_seq_dim];
    const uint block_end_pos = blocked_indexes_end[target_seq_dim];
    const uint seq_idx_end = block_end_pos - block_start_pos;
#else
    const uint seq_idx_end = min(TARGET_SEQ_LEN - target_seq_idx, (uint)TARGET_SEQ_LEN_BLOCK_SIZE);
#endif
    const uint num_read_blocks = K_HEAD_SIZE == V_HEAD_SIZE ? 1 :  CEIL_DIV(K_HEAD_SIZE, V_HEAD_SIZE);

    for (int read_blk_idx = 0; read_blk_idx < num_read_blocks; read_blk_idx++)
    {
        uint k_head_size_idx = (read_blk_idx * V_HEAD_SIZE * SG_SCALE_FACTOR + get_local_id(2)) % K_HEAD_SIZE;
        uint k_sgid =  read_blk_idx * K_HEAD_SIZE * SG_SCALE_FACTOR / TARGET_SEQ_LEN_BLOCK_SIZE / num_read_blocks + sgid;
        // Load Q input to SLM and transpose it
#if IS_PAGED_ATTENTION
        uint query_offset = INPUT0_OFFSET +
                            block_start_pos * (K_HEAD_SIZE * NUM_HEADS + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM) +
                            num_heads_dim * K_HEAD_SIZE + k_head_size_idx;
        const uint query_pitch = (K_HEAD_SIZE * NUM_HEADS + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM);
#else
#ifdef INPUT0_DIMS_ORDER
        uint query_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx, (k_head_size_idx));
        uint query_offset_next_seq = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, target_seq_idx + 1, (k_head_size_idx));
        const uint query_pitch = query_offset_next_seq - query_offset;
#else
        uint query_offset = INPUT0_GET_INDEX(b0_idx, b1_idx, target_seq_idx, (k_head_size_idx));

        const uint query_pitch = K_HEAD_SIZE;
#endif
#endif
        uint query_local_offset = k_head_size_idx * TARGET_SEQ_LEN_BLOCK_SIZE;

#if APPLY_SCALES_TO_QUERY
#if HAS_SCALE_INPUT
        const INPUT0_TYPE scale_val = *scale;
#else
        const INPUT0_TYPE scale_val = TO_INPUT0_TYPE(STATIC_SCALE_VALUE);
#endif
#else
        const INPUT0_TYPE scale_val = INPUT0_VAL_ONE;
#endif

        if (seq_idx_end != TARGET_SEQ_LEN_BLOCK_SIZE) {
            #ifdef K_HEAD_SIZE_LEFTOVER
            if ((k_sgid + 1) * SUBGROUP_SIZE <= K_HEAD_SIZE) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                    slm_query[query_local_offset] = val * scale_val;
                    query_offset += query_pitch;
                    query_local_offset++;
                }
            } else {
                // remainder
                int valid_workers = K_HEAD_SIZE - k_sgid * SUBGROUP_SIZE;
                if (sglid < valid_workers) {
                    unroll_for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                        INPUT0_TYPE val = query_input[query_offset];
                        slm_query[query_local_offset] = val * scale_val;
                        query_offset += query_pitch;
                        query_local_offset++;
                    }
                }
            }
            #else
            if (k_sgid * SUBGROUP_SIZE < K_HEAD_SIZE) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);
                    slm_query[query_local_offset] = val * scale_val;
                    query_offset += query_pitch;
                    query_local_offset++;
                }
            }
            #endif
        } else {
            #if SG_SCALE_FACTOR == 2
                if (k_sgid < (K_HEAD_SIZE / SUBGROUP_SIZE)) {
                    unroll_for (uint seq_idx = 0; seq_idx < (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR); seq_idx++) {
                        INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);
                        slm_query[query_local_offset] = val * scale_val;
                        query_offset += query_pitch;
                        query_local_offset++;
                    }
                } else {
                    query_local_offset += (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR);
                    query_offset += query_pitch * (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR);
                    unroll_for (uint seq_idx = 0; seq_idx < (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR); seq_idx++) {
                        INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                        slm_query[query_local_offset] = val * scale_val;
                        query_offset += query_pitch;
                        query_local_offset++;
                    }
                }
            #elif SG_SCALE_FACTOR == 4
                query_local_offset += (k_sgid / (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) * (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR);
                query_offset += query_pitch * (k_sgid / (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) * (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR);
                unroll_for (uint seq_idx = 0; seq_idx < (TARGET_SEQ_LEN_BLOCK_SIZE / SG_SCALE_FACTOR); seq_idx++) {
                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                    slm_query[query_local_offset] = val * scale_val;
                    query_offset += query_pitch;
                    query_local_offset++;
                }
            #else
            // Load query_input to slm_query. E.g., if TARGET_SEQ_LEN_BLOCK_SIZE = 16 && K_HEAD_SIZE == 72 && # subgroups = 5
            //     16 (TARGET_SEQ_LEN_BLOCK_SIZE)
            // ------------
            // |   sg0     | 16
            // |-----------|
            // |   sg1     | 16
            // |-----------|
            // |   sg2     | 16
            // |-----------|
            // |   sg3     | 16
            // |-----------|
            // |   sg4     | 8 // remainder of head
            // -------------
            //    slm_query
            if ((k_sgid + 1) * TARGET_SEQ_LEN_BLOCK_SIZE <= K_HEAD_SIZE) {
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                    INPUT0_TYPE val = BLOCK_READN(INPUT0_TYPE, 1, query_input, query_offset);

                    slm_query[query_local_offset] = val * scale_val;
                    query_offset += query_pitch;
                    query_local_offset++;
                }
            } else {
                // remainder
                int valid_workers = K_HEAD_SIZE - k_sgid * SUBGROUP_SIZE;
                if (sglid < valid_workers) {
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        INPUT0_TYPE val = query_input[query_offset];
                        slm_query[query_local_offset] = val * scale_val;
                        query_offset += query_pitch;
                        query_local_offset++;
                    }
                }
            }
            #endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    {
        #if TARGET_SEQ_LEN_BLOCK_SIZE <= SUBGROUP_SIZE
            // Initialize slm buffers with MIN and ZERO values
            if (sgid == 0 && sglid < TARGET_SEQ_LEN_BLOCK_SIZE) {
                slm_max_val_prev[sglid] = SOFTMAX_ACCUMULATOR_VAL_MIN;
                slm_exp_sum_prev[sglid] = SOFTMAX_ACCUMULATOR_VAL_ZERO;
            }
        #else
            #error sdpa_opt.cl: unsupported TARGET_SEQ_LEN_BLOCK_SIZE
        #endif
    }

    // Q*K calculation loop
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) output_acc = OUTPUT_VAL_ZERO;

    __attribute__((opencl_unroll_hint(1)))
    for (uint start_partition_idx = 0; start_partition_idx < SOURCE_SEQ_LEN; start_partition_idx += SEQ_LEN_PARTITION_SIZE) {
        const uint seq_len = start_partition_idx + sgid * SUBGROUP_SIZE;
#if IS_CAUSAL
        const uint partition_seq_len = min((uint)SEQ_LEN_PARTITION_SIZE, (uint)max(0, (int)(target_seq_idx + seq_idx_end) - (int)start_partition_idx));
#else
        const uint partition_seq_len = min((uint)SOURCE_SEQ_LEN - start_partition_idx, (uint)SEQ_LEN_PARTITION_SIZE);
#endif

        MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_acc = INPUT0_VAL_ZERO;
#if IS_CAUSAL
        if (seq_len <= target_seq_idx) { // keep tril i.e. m >= n
#endif
#if IS_PAGED_ATTENTION
#ifdef BROADCAST_GROUP_SIZE
        const uint heads_dim = num_heads_dim / BROADCAST_GROUP_SIZE;
#else
        const uint heads_dim = num_heads_dim;
#endif
        #define KEY_SEQ_OFFSET subsequence_begins[gws_seq_indexes_correspondence[target_seq_dim]]
        const uint key_pitch = (K_HEAD_SIZE * NUM_KV_HEADS + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM);
        uint key_offset = INPUT1_OFFSET +
                          KEY_SEQ_OFFSET * key_pitch +
                          heads_dim * K_HEAD_SIZE +
                          seq_len * key_pitch;
#else // !IS_PAGED_ATTENTION
#ifdef BEAM_TABLE_TYPE
            const uint b_idx = beam_table[FUNC_CALL(get_bt_index_key)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len + sglid, 0)];
            const uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, seq_len + sglid, 0);
#else
            const uint b_idx = b0_idx;
    #ifdef INPUT1_DIMS_ORDER
            uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len, 0);
            uint key_offset_next_seq = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, seq_len + 1, 0);
            const uint key_pitch = key_offset_next_seq - key_offset;
    #else
            uint key_offset = INPUT1_GET_INDEX(b0_idx, b1_idx, seq_len, 0);
            const uint key_pitch = K_HEAD_SIZE;
    #endif
#endif // BEAM_TABLE_TYPE
#endif // IS_PAGED_ATTENTION
            int seq_len_calc_size = min((int)(SOURCE_SEQ_LEN) - (int)seq_len, (int)SUBGROUP_SIZE);
#if !IS_CAUSAL
            qk_acc = FUNC_CALL(load_attn_mask)(OPTIONAL_SHAPE_INFO_TENSOR
                            b0_idx,
                            b1_idx,
                            target_seq_idx + sglid,
                            // TODO: pass seq_len_calc_size here
                            seq_len
                            ATTN_MASK_BUFFER
                            ATTN_SCALE_BUFFER
                            PA_BUFFERS);
#endif  // !IS_CAUSAL

            if (seq_len_calc_size >= SUBGROUP_SIZE) {
#if IS_KV_COMPRESSED
                const uint comp_offset = GET_COMPRESSION_INDEX(KEY_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, seq_len + sglid, 0);
                KEY_COMPRESSION_SCALE_TYPE comp_scale = key_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                KEY_COMPRESSION_SCALE_TYPE comp_zp = key_scale[comp_offset + 1];
#endif
#endif
                uint head_idx_index = 0;
                __attribute__((opencl_unroll_hint(1)))
                for (; head_idx_index + SUBGROUP_SIZE <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset);
                    #define QUERY_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)

                    QUERY_VEC queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; q_row_idx++) {
                        queries_vec[q_row_idx] = slm_query[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }

                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; key_row_idx++) {
#ifdef BEAM_TABLE_TYPE
                        const INPUT1_TYPE key_packed = KEY_BLOCK_READ(key_input, sub_group_broadcast(key_offset, key_row_idx) + head_idx_index);
#else
                        const INPUT1_TYPE key_packed = KEY_BLOCK_READ(key_input, key_offset + key_row_idx * key_pitch + head_idx_index);
#endif

#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        KEY_COMPRESSION_SCALE_TYPE key_vals = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) - sub_group_broadcast(comp_zp, key_row_idx)) * sub_group_broadcast(comp_scale, key_row_idx);
#elif IS_KV_COMPRESSED
                        KEY_COMPRESSION_SCALE_TYPE key_vals = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) * sub_group_broadcast(comp_scale, key_row_idx));
#else
                        INPUT1_TYPE key_vals = key_packed;
#endif

                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                }
                #ifdef K_HEAD_SIZE_LEFTOVER
                    QUERY_VEC queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < K_HEAD_SIZE_LEFTOVER; q_row_idx++) {
                        queries_vec[q_row_idx] = slm_query[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }
                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; ++key_row_idx) {
#ifdef BEAM_TABLE_TYPE
                        INPUT1_TYPE key_packed = (sglid < K_HEAD_SIZE_LEFTOVER) ? key_input[sub_group_broadcast(key_offset, key_row_idx) + head_idx_index + sglid] : INPUT1_VAL_ZERO;
#else
                        INPUT1_TYPE key_packed = (sglid < K_HEAD_SIZE_LEFTOVER) ? key_input[key_offset + key_row_idx * key_pitch + head_idx_index + sglid] : INPUT1_VAL_ZERO;
#endif
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        KEY_COMPRESSION_SCALE_TYPE key_vals = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) - sub_group_broadcast(comp_zp, key_row_idx)) * sub_group_broadcast(comp_scale, key_row_idx);j
#elif IS_KV_COMPRESSED
                        KEY_COMPRESSION_SCALE_TYPE key_vals = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) * sub_group_broadcast(comp_scale, key_row_idx));
#else
                        INPUT1_TYPE key_vals = key_packed;
#endif
                        unroll_for (uint i = 0; i < K_HEAD_SIZE_LEFTOVER; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                #endif
            } else if (seq_len_calc_size > 0) {
#if IS_KV_COMPRESSED
                const uint comp_offset = GET_COMPRESSION_INDEX(KEY_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, seq_len + min(sglid, (uint)seq_len_calc_size - 1), 0);
                // const uint comp_offset = GET_COMPRESSION_INDEX(KEY_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, seq_len + sglid, 0);
                KEY_COMPRESSION_SCALE_TYPE comp_scale = key_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                KEY_COMPRESSION_SCALE_TYPE comp_zp = key_scale[comp_offset + 1];
#endif
#endif
                uint head_idx_index = 0;
                __attribute__((opencl_unroll_hint(1)))
                for (; head_idx_index + SUBGROUP_SIZE <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE) {
                    #define KEY_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
                    #define QUERY_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)
#if IS_KV_COMPRESSED
                    #define KEY_UNPACKED_TYPE KEY_COMPRESSION_SCALE_TYPE
                    #define KEY_UNPACKED_VEC_TYPE MAKE_VECTOR_TYPE(KEY_COMPRESSION_SCALE_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)
                    #define TO_KEY_UNPACKED_TYPE(val) TO_KEY_COMPRESSION_SCALE_TYPE(val)
#else
                    #define KEY_UNPACKED_TYPE INPUT1_TYPE
                    #define KEY_UNPACKED_VEC_TYPE MAKE_VECTOR_TYPE(INPUT1_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE)
                    #define TO_KEY_UNPACKED_TYPE(val) TO_INPUT1_TYPE(val)
#endif

                    QUERY_VEC_TYPE queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; q_row_idx++) {
                        queries_vec[q_row_idx] = slm_query[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }

#ifndef LOAD_KEY_LEFTOVERS_IN_CALC_LOOP
                    KEY_UNPACKED_VEC_TYPE key_vec = 0;
                    unroll_for (uint key_row_idx = 0; key_row_idx < seq_len_calc_size; key_row_idx++) {
#ifdef BEAM_TABLE_TYPE
                        key_vec[key_row_idx] = TO_KEY_UNPACKED_TYPE(KEY_BLOCK_READ(key_input, sub_group_broadcast(key_offset, key_row_idx) + head_idx_index));
#else
                        key_vec[key_row_idx] = TO_KEY_UNPACKED_TYPE(KEY_BLOCK_READ(key_input, key_offset + key_row_idx * key_pitch + head_idx_index));
#endif

#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        key_vec[key_row_idx] = (key_vec[key_row_idx] - sub_group_broadcast(comp_zp, key_row_idx)) * sub_group_broadcast(comp_scale, key_row_idx);
#elif IS_KV_COMPRESSED
                        key_vec[key_row_idx] *= sub_group_broadcast(comp_scale, key_row_idx);
#endif
                    }
#endif // LOAD_KEY_LEFTOVERS_IN_CALC_LOOP

                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; key_row_idx++) {
#ifdef LOAD_KEY_LEFTOVERS_IN_CALC_LOOP
                        KEY_UNPACKED_TYPE key_vals = 0;
                        if (key_row_idx < seq_len_calc_size) {
#ifdef BEAM_TABLE_TYPE
                            key_vals = TO_KEY_UNPACKED_TYPE(KEY_BLOCK_READ(key_input, sub_group_broadcast(key_offset, key_row_idx) + head_idx_index));
#else
                            key_vals = TO_KEY_UNPACKED_TYPE(KEY_BLOCK_READ(key_input, key_offset + key_row_idx * key_pitch + head_idx_index));
#endif // BEAM_TABLE_TYPE
                        }
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                            key_vals = (key_vals - sub_group_broadcast(comp_zp, key_row_idx)) * sub_group_broadcast(comp_scale, key_row_idx);
#elif IS_KV_COMPRESSED
                            key_vals *= sub_group_broadcast(comp_scale, key_row_idx);
#endif
#else   // !defined(LOAD_KEY_LEFTOVERS_IN_CALC_LOOP)
                        #define key_vals key_vec[key_row_idx]
#endif  // !defined(LOAD_KEY_LEFTOVERS_IN_CALC_LOOP)
                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_vals, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                }
                #ifdef K_HEAD_SIZE_LEFTOVER
                    QUERY_VEC queries_vec;
                    uint query_local_offset = (head_idx_index * TARGET_SEQ_LEN_BLOCK_SIZE) + sglid;
                    unroll_for (uint q_row_idx = 0; q_row_idx < K_HEAD_SIZE_LEFTOVER; q_row_idx++) {
                        queries_vec[q_row_idx] = slm_query[query_local_offset];
                        query_local_offset += TARGET_SEQ_LEN_BLOCK_SIZE;
                    }
                    unroll_for (uint key_row_idx = 0; key_row_idx < TARGET_SEQ_LEN_BLOCK_SIZE; ++key_row_idx) {
#ifdef BEAM_TABLE_TYPE
                        const INPUT1_TYPE key_packed = (sglid < K_HEAD_SIZE_LEFTOVER) ? key_input[sub_group_broadcast(key_offset, key_row_idx) + head_idx_index + sglid] : INPUT1_VAL_ZERO;
#else
                        const INPUT1_TYPE key_packed = (sglid < K_HEAD_SIZE_LEFTOVER) ? key_input[key_offset + key_row_idx * key_pitch + head_idx_index + sglid] : INPUT1_VAL_ZERO;
#endif
#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        KEY_COMPRESSION_SCALE_TYPE key_val = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) - sub_group_broadcast(comp_zp, key_row_idx)) * sub_group_broadcast(comp_scale, key_row_idx);j
#elif IS_KV_COMPRESSED
                        KEY_COMPRESSION_SCALE_TYPE key_val = (TO_KEY_COMPRESSION_SCALE_TYPE(key_packed) * sub_group_broadcast(comp_scale, key_row_idx));
#else
                        INPUT1_TYPE key_val = key_packed;
#endif
                        unroll_for (uint i = 0; i < K_HEAD_SIZE_LEFTOVER; i++) {
                            qk_acc[key_row_idx] = mad(sub_group_broadcast(key_val, i), queries_vec[i], qk_acc[key_row_idx]);
                        }
                    }
                #endif // K_HEAD_SIZE_LEFTOVER
            }

            // softmax_scale
            {
                SOFTMAX_ACCUMULATOR_TYPE qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;
                unroll_for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
#if IS_CAUSAL
                    // casual mask: valid only if m >= n
#if defined(IS_PAGED_ATTENTION) && SLIDING_WINDOW_SIZE != 0
                    if ((seq_len + i <= target_seq_idx + sglid) && (target_seq_idx + sglid < SLIDING_WINDOW_SIZE || seq_len + i > target_seq_idx + sglid - SLIDING_WINDOW_SIZE)) {
#else
                    if (seq_len + i <= target_seq_idx + sglid) {
#endif
#endif  // IS_CAUSAL
#if !APPLY_SCALES_TO_QUERY
#if HAS_SCALE_INPUT
                        const OUTPUT_TYPE scale_val = *scale;
#else
                        const OUTPUT_TYPE scale_val = TO_OUTPUT_TYPE(STATIC_SCALE_VALUE);
#endif
                        qk_acc[i] *= scale_val;
#endif // !APPLY_SCALES_TO_QUERY

#ifdef HAS_ALIBI
                        const int alibi_val = (1 - SOURCE_SEQ_LEN) + seq_len + i;
                        qk_acc[i] += alibi_slopes[num_heads_dim] * alibi_val;
#endif

                        qk_acc[i] = INPUT0_MIN_FUNC(INPUT0_MAX_FUNC(qk_acc[i], INPUT0_VAL_MIN), INPUT0_VAL_MAX);
#if IS_CAUSAL
                    } else {
                        qk_acc[i] = INPUT0_VAL_MIN;
                    }
#endif  // IS_CAUSAL
                    qk_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max, TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc[i]));
#if IS_FLASHATTEN_V2
                    slm_qk_vals[sglid][sgid * TARGET_SEQ_LEN_BLOCK_SIZE + i] = qk_acc[i];
#endif
                } // unroll_for
                {
#if IS_FLASHATTEN_V2
                    slm_qk_max_vals[sglid][sgid] = qk_max;
#else
                    slm_qk_max_vals[sgid][sglid] = qk_max;
                    qk_max = SOFTMAX_ACCUMULATOR_VAL_MIN;  // sounds no need to reset?
#endif
                }
            } // end of softmax_scale
#if IS_CAUSAL
        } else { // skip triu
#if IS_FLASHATTEN_V2
            slm_qk_max_vals[sglid][sgid] = SOFTMAX_ACCUMULATOR_VAL_MIN;
#else
            slm_qk_max_vals[sgid][sglid] = SOFTMAX_ACCUMULATOR_VAL_MIN;
#endif
        }
#endif // IS_CAUSAL

        barrier(CLK_LOCAL_MEM_FENCE);

        // SoftMax calculation
#if IS_FLASHATTEN_V2
        {
            // each sg will compute a whole row of query
            uint aligned_width = ((SUBGROUPS_PER_WG + (SUBGROUP_SIZE-1)) & ~(SUBGROUP_SIZE-1));
            for (uint m = sgid; m < seq_idx_end; m += SUBGROUPS_PER_WG) {
                SOFTMAX_ACCUMULATOR_TYPE qk_max_new, qk_max_cur = SOFTMAX_ACCUMULATOR_VAL_MIN;
                for (uint k = sglid; k <  aligned_width; k += SUBGROUP_SIZE) {
                    if (k < SUBGROUPS_PER_WG) {
                        qk_max_new = slm_qk_max_vals[m][k];
                    } else {
                        qk_max_new = SOFTMAX_ACCUMULATOR_VAL_MIN;
                    }
                    qk_max_new = SOFTMAX_ACCUMULATOR_MAX_FUNC(sub_group_reduce_max(qk_max_new), qk_max_cur);
                    qk_max_cur = qk_max_new;
                }

                SOFTMAX_ACCUMULATOR_TYPE max_val_prev = slm_max_val_prev[m];
                qk_max_new = SOFTMAX_ACCUMULATOR_MAX_FUNC(sub_group_reduce_max(qk_max_cur), max_val_prev);

                // softmax
                SOFTMAX_ACCUMULATOR_TYPE exp_sum_new = SOFTMAX_ACCUMULATOR_VAL_ZERO;
                for (uint k = sglid; k < partition_seq_len; k += SUBGROUP_SIZE) {
                    SOFTMAX_ACCUMULATOR_TYPE a = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(slm_qk_vals[m][k]) - qk_max_new);
                    slm_qk_vals[m][k] = TO_OUTPUT_TYPE(a);
                    exp_sum_new += a;
                }
                exp_sum_new = sub_group_reduce_add(exp_sum_new);

#if PAGED_ATTENTION_SCORES_OUTPUT
                {
                    const uint subsequence_idx = gws_seq_indexes_correspondence[target_seq_dim];
                    const uint subsequence_end_pos = subsequence_begins[subsequence_idx + 1];

                    // PagedAttention is supposed to save last N "rows" of the QK matrix multiplication,
                    // so save SEQ_LEN_PARTITION_SIZE * N elements for each partition
#if IS_PAGED_ATTENTION && HAS_SCORE_AGGREGATION
                    const int scores_offset = cumulative_score_aggregation_sum[subsequence_idx];
                    const int window_size = cumulative_score_aggregation_sum[subsequence_idx + 1] - scores_offset;
                    const int global_first_row_idx = max((int)subsequence_end_pos - window_size, 0);
                    if (subsequence_end_pos > global_first_row_idx) {
                        const bool save_row = m >= max(global_first_row_idx - (int)block_start_pos, 0);
                        const int local_row_idx = block_start_pos + m - global_first_row_idx;
#else
                    const int scores_offset = subsequence_idx;
                    if (subsequence_end_pos == block_end_pos) {
                        const bool save_row = m == (block_end_pos - block_start_pos - 1);
                        const int local_row_idx = 0;
#endif
                        if (save_row) {
                            const uint partition_idx = start_partition_idx / SEQ_LEN_PARTITION_SIZE;

                            SOFTMAX_ACCUMULATOR_TYPE correction_factor = native_exp(qk_max_new - qk_max_cur);

                            if (sglid == 0) {
                                const uint max_partitions_num = aligned_max_context_len / SEQ_LEN_PARTITION_SIZE;
                                const uint exp_sums_output_offset = (scores_offset + local_row_idx) * NUM_HEADS * max_partitions_num +
                                                                    num_heads_dim * max_partitions_num +
                                                                    partition_idx;
                                exp_sums[exp_sums_output_offset] = exp_sum_new * correction_factor;
                                max_logits[exp_sums_output_offset] = qk_max_cur;
                            }

                            const uint output_offset = (scores_offset + local_row_idx) * NUM_HEADS * aligned_max_context_len +
                                                       num_heads_dim * aligned_max_context_len +
                                                       partition_idx * SEQ_LEN_PARTITION_SIZE;
                            for (uint i = sglid; i < partition_seq_len; i += SUBGROUP_SIZE) {
                                softmax_results[output_offset + i] = TO_SOFTMAX_ACCUMULATOR_TYPE(slm_qk_vals[m][i]) / exp_sum_new;
                            }
#if HAS_SCORE_AGGREGATION
                            const uint full_partition_seq_len = min((uint)SOURCE_SEQ_LEN - start_partition_idx, (uint)SEQ_LEN_PARTITION_SIZE);
                            for (uint i = partition_seq_len + sglid; i < full_partition_seq_len; i += SUBGROUP_SIZE) {
                                softmax_results[output_offset + i] = SOFTMAX_ACCUMULATOR_VAL_ZERO;
                            }
#endif
                        }
                    }
                }
#endif /*end of PAGED_ATTENTION_SCORES_OUTPUT*/

                // update
                if (sglid == 0) {
                    SOFTMAX_ACCUMULATOR_TYPE pre_exp_sum = slm_exp_sum_prev[m];
                    SOFTMAX_ACCUMULATOR_TYPE correction_factor = native_exp(max_val_prev - qk_max_new);
                    SOFTMAX_ACCUMULATOR_TYPE pre_exp_sum_fixed = pre_exp_sum * correction_factor;
                    exp_sum_new += pre_exp_sum_fixed;

                    slm_update_factor[m] = correction_factor;
                    slm_max_val_prev[m] = qk_max_new;
                    slm_exp_sum_prev[m] = exp_sum_new;
                }
            }
        }
#else /*!IS_FLASHATTEN_V2*/
        {
            SOFTMAX_ACCUMULATOR_TYPE qk_max_new = SOFTMAX_ACCUMULATOR_VAL_MIN;

            for (uint i = 0; i < SUBGROUPS_PER_WG; i++) {
                SOFTMAX_ACCUMULATOR_TYPE qk_max_val = slm_qk_max_vals[i][sglid];
                qk_max_new = SOFTMAX_ACCUMULATOR_MAX_FUNC(qk_max_new, qk_max_val);
            }

            if (sgid == 0) {
                slm_max_val_cur[sglid] = qk_max_new;
            }

            SOFTMAX_ACCUMULATOR_TYPE exp_sum_new = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                qk_acc[i] = native_exp(TO_SOFTMAX_ACCUMULATOR_TYPE(qk_acc[i]) - qk_max_new);
#if IS_CAUSAL
#if defined(IS_PAGED_ATTENTION) && SLIDING_WINDOW_SIZE != 0
                if ((seq_len + i <= target_seq_idx + sglid) && (target_seq_idx + sglid < SLIDING_WINDOW_SIZE || seq_len + i >= target_seq_idx + sglid - SLIDING_WINDOW_SIZE)) {
#else
                if (seq_len + i <= target_seq_idx + sglid) {
#endif
                    exp_sum_new += qk_acc[i];
                }
# else
                exp_sum_new += qk_acc[i];
#endif
            }

            {
                slm_exp_sum_vals[sgid * SUBGROUP_SIZE + sglid] = exp_sum_new;
            }

            exp_sum_new = SOFTMAX_ACCUMULATOR_VAL_ZERO;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < SUBGROUPS_PER_WG; i++) {
                SOFTMAX_ACCUMULATOR_TYPE exp_sum = slm_exp_sum_vals[i * SUBGROUP_SIZE + sglid];
                exp_sum_new += exp_sum;
            }

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                qk_acc[i] = qk_acc[i] / exp_sum_new;
            }

            if (sgid == 0) {
                slm_exp_sum_cur[sglid] = exp_sum_new;
            }

            for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
                slm_qk_vals[sglid][sgid * TARGET_SEQ_LEN_BLOCK_SIZE + i] = qk_acc[i];
            }

#if PAGED_ATTENTION_SCORES_OUTPUT
            {
                const uint subsequence_idx = gws_seq_indexes_correspondence[target_seq_dim];
                const uint subsequence_end_pos = subsequence_begins[subsequence_idx + 1];

                // PagedAttention is supposed to save only last "row" of the QK matrix multiplication,
                // so save SEQ_LEN_PARTITION_SIZE elements for each partition
#if IS_PAGED_ATTENTION && HAS_SCORE_AGGREGATION
                const int scores_offset = cumulative_score_aggregation_sum[subsequence_idx];
                const int window_size = cumulative_score_aggregation_sum[subsequence_idx + 1] - scores_offset;
                const int global_first_row_idx = max((int)subsequence_end_pos - window_size, 0);
                if (subsequence_end_pos > global_first_row_idx) {
                    const int local_row_idx = block_start_pos + sglid - global_first_row_idx;
                    const bool save_row = sglid >= max(global_first_row_idx - (int)block_start_pos, 0) && local_row_idx < window_size;
#else
                const int scores_offset = subsequence_idx;
                if (subsequence_end_pos == block_end_pos) {
                    const int local_row_idx = 0;
                    const bool save_row = sglid == block_end_pos - block_start_pos - 1;
#endif
                    const uint last_row_idx = block_end_pos - block_start_pos - 1;
                    if (save_row) {
                        const uint partition_idx = start_partition_idx / SEQ_LEN_PARTITION_SIZE;

                        if (sgid == 0) {
                            const uint max_partitions_num = aligned_max_context_len / SEQ_LEN_PARTITION_SIZE;
                            const uint exp_sums_output_offset = (scores_offset + local_row_idx) * NUM_HEADS * max_partitions_num +
                                                                num_heads_dim * max_partitions_num +
                                                                partition_idx;
                            exp_sums[exp_sums_output_offset] = exp_sum_new;
                            max_logits[exp_sums_output_offset] = qk_max_new;
                        }

                        const uint output_offset = (scores_offset + local_row_idx) * NUM_HEADS * aligned_max_context_len +
                                                   num_heads_dim * aligned_max_context_len +
                                                   partition_idx * SEQ_LEN_PARTITION_SIZE + sgid * TARGET_SEQ_LEN_BLOCK_SIZE;
                        for (uint i = 0; i < TARGET_SEQ_LEN_BLOCK_SIZE; i++) {
#if HAS_SCORE_AGGREGATION
                            softmax_results[output_offset + i] = sgid * TARGET_SEQ_LEN_BLOCK_SIZE + i >= partition_seq_len ? SOFTMAX_ACCUMULATOR_VAL_ZERO : qk_acc[i];
#else
                            softmax_results[output_offset + i] = qk_acc[i];
#endif
                        }
                    }
                }
            }
#endif  /*end of PAGED_ATTENTION_SCORES_OUTPUT*/
        }
#endif /*end of softmax calc*/

        barrier(CLK_LOCAL_MEM_FENCE);

        // QK*V calculation
        {
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) acc_output_res = OUTPUT_VAL_ZERO;
#if IS_PAGED_ATTENTION
            const uint value_pitch = (V_HEAD_SIZE * NUM_KV_HEADS + INPUT2_PAD_BEFORE_FEATURE_NUM + INPUT2_PAD_AFTER_FEATURE_NUM);
#else
#ifdef INPUT2_DIMS_ORDER
            uint value_offset_base = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 0, 0);
            uint value_offset_next_seq = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, 1, 0);
            const uint value_pitch = value_offset_next_seq - value_offset_base;
#else
            const uint value_pitch = V_HEAD_SIZE;
#endif
#endif

            if (partition_seq_len == SEQ_LEN_PARTITION_SIZE) {
                uint seq_len_start = (sgid / (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) * (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR);
                for (uint seq_len = seq_len_start; seq_len < seq_len_start + (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR); seq_len += SUBGROUP_SIZE) {
#if IS_PAGED_ATTENTION
#ifdef BROADCAST_GROUP_SIZE
                    const uint heads_dim = num_heads_dim / BROADCAST_GROUP_SIZE;
#else
                    const uint heads_dim = num_heads_dim;
#endif
                    const uint value_seq_offset = subsequence_begins[gws_seq_indexes_correspondence[target_seq_dim]];
                    uint value_offset = INPUT2_OFFSET +
                                        value_seq_offset * value_pitch +
                                        heads_dim * V_HEAD_SIZE +
                                        (start_partition_idx + (seq_len)) * value_pitch + head_size_idx;
#else
#ifdef BEAM_TABLE_TYPE
                    const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len) + sglid, sgid * SUBGROUP_SIZE)];
                    const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len) + sglid, sgid * SUBGROUP_SIZE);
#else
                    const uint b_idx = b0_idx;
    #ifdef INPUT2_DIMS_ORDER
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len), head_size_idx);
    #else
                    uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + (seq_len), head_size_idx);
    #endif
#endif
#endif
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_val;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        qk_val[seq_idx] = slm_qk_vals[seq_idx][seq_len + sglid];
                    }
#if IS_KV_COMPRESSED
                    const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + seq_len + sglid, 0);
                    VALUE_COMPRESSION_SCALE_TYPE comp_scale = val_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                    VALUE_COMPRESSION_SCALE_TYPE comp_zp = val_scale[comp_offset + 1];
#endif
#endif
                    #ifdef V_HEAD_SIZE_LEFTOVER
                    // splitting to two cases for supressing reg spill : block_read & eltwise read
                    if (sgid < SUBGROUPS_PER_WG - 1) {
                    // block_read values
                    #endif
                    unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                        #ifdef BEAM_TABLE_TYPE
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, i));
                        #else
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, value_offset);
                        #endif

                        #if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
                        #elif IS_KV_COMPRESSED
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, i));
                        #else
                        INPUT2_TYPE value_val = value_packed;
                        #endif
                        unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                            acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
                        }
                        #ifndef BEAM_TABLE_TYPE
                        value_offset += value_pitch;
                        #endif
                    } // unroll_for
                    #ifdef V_HEAD_SIZE_LEFTOVER
                    } else if (head_size_idx < V_HEAD_SIZE) {
                    // read values element by element
                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                        #ifdef BEAM_TABLE_TYPE
                            const INPUT2_TYPE value_packed = value_input[sub_group_broadcast(value_offset, i)];
                        #else
                            const INPUT2_TYPE value_packed = value_input[value_offset];
                        #endif
                        #if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                            VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
                        #elif IS_KV_COMPRESSED
                            VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, i));
                        #else
                            INPUT2_TYPE value_val = value_packed;
                        #endif
                            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                                acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
                            }
                        #ifndef BEAM_TABLE_TYPE
                            value_offset += value_pitch;
                        #endif
                        }
                    } // head_size_idx < V_HEAD_SIZE && sgid < SUBGROUPS_PER_WG - 1
                    #endif // V_HEAD_SIZE_LEFTOVER
                } // for seq_len
            } else { // partition_seq_len is less than SEQ_LEN_PARTITION_SIZE
                const uint seq_len_start = (sgid / (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) * (SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR);
                uint seq_len_end = 0;
                if (seq_len_start < partition_seq_len)
                    seq_len_end = seq_len_start + min(partition_seq_len - seq_len_start, (uint)(SEQ_LEN_PARTITION_SIZE / SG_SCALE_FACTOR));;
                for (uint seq_len = seq_len_start / SUBGROUP_SIZE; seq_len < seq_len_end / SUBGROUP_SIZE; seq_len++) {
            #if IS_PAGED_ATTENTION
                #ifdef BROADCAST_GROUP_SIZE
                    const uint heads_dim = num_heads_dim / BROADCAST_GROUP_SIZE;
                #else
                    const uint heads_dim = num_heads_dim;
                #endif
                    const uint value_seq_offset = subsequence_begins[gws_seq_indexes_correspondence[target_seq_dim]];
                    uint value_offset = INPUT2_OFFSET +
                                        value_seq_offset * value_pitch +
                                        heads_dim * V_HEAD_SIZE +
                                        (start_partition_idx + (seq_len * SUBGROUP_SIZE)) * value_pitch + head_size_idx;
            #else // !IS_PAGED_ATTENTION
                #ifdef BEAM_TABLE_TYPE
                    const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE)];
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, sgid * SUBGROUP_SIZE);
                #else
                    const uint b_idx = b0_idx;
                #ifdef INPUT2_DIMS_ORDER
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
                #else
                    uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + (seq_len * SUBGROUP_SIZE), head_size_idx);
                #endif
            #endif
#endif

#if IS_KV_COMPRESSED
                    const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + (seq_len * SUBGROUP_SIZE) + sglid, 0);
                    VALUE_COMPRESSION_SCALE_TYPE comp_scale = val_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                    VALUE_COMPRESSION_SCALE_TYPE comp_zp = val_scale[comp_offset + 1];
#endif
#endif // IS_KV_COMPRESSED

                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_val;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        qk_val[seq_idx] = slm_qk_vals[seq_idx][seq_len * SUBGROUP_SIZE + sglid];
                    }
                #ifdef V_HEAD_SIZE_LEFTOVER
                    // splitting to two cases for supressing reg spill : block_read & eltwise read
                    if (sgid < SUBGROUPS_PER_WG - 1) {
                    // block_read values
                #endif
                    unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                #ifdef BEAM_TABLE_TYPE
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, i));
                #else
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, value_offset);
                #endif

                #if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
                #elif IS_KV_COMPRESSED
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, i));
                #else
                        INPUT2_TYPE value_val = value_packed;
                #endif
                        unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                            acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
                        }
                #ifndef BEAM_TABLE_TYPE
                        value_offset += value_pitch;
                #endif
                    } // unroll_for
                #ifdef V_HEAD_SIZE_LEFTOVER
                    } else if (head_size_idx < V_HEAD_SIZE) {
                    // read values element by element
                        unroll_for (uint i = 0; i < SUBGROUP_SIZE; i++) {
                        #ifdef BEAM_TABLE_TYPE
                            const INPUT2_TYPE value_packed = value_input[sub_group_broadcast(value_offset, i)];
                        #else
                            const INPUT2_TYPE value_packed = value_input[value_offset];
                        #endif
                        #if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                            VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, i)) * sub_group_broadcast(comp_scale, i);
                        #elif IS_KV_COMPRESSED
                            VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, i));
                        #else
                            INPUT2_TYPE value_val = value_packed;
                        #endif
                            unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                                acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], i), value_val, acc_output_res[seq_idx]);
                            }
                        #ifndef BEAM_TABLE_TYPE
                            value_offset += value_pitch;
                        #endif
                        }
                    }
                #endif // V_HEAD_SIZE_LEFTOVER
                } // for seq_len

                // QK*V leftovers processing
                const uint seq_len_leftovers_start = ((seq_len_end / SUBGROUP_SIZE) * SUBGROUP_SIZE);
                if (seq_len_leftovers_start != seq_len_end) {
                    MAKE_VECTOR_TYPE(OUTPUT_TYPE, TARGET_SEQ_LEN_BLOCK_SIZE) qk_val;
                    unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                        qk_val[seq_idx] = slm_qk_vals[seq_idx][seq_len_leftovers_start+sglid];
                    }
#if IS_PAGED_ATTENTION
#ifdef BROADCAST_GROUP_SIZE
                    const uint heads_dim = num_heads_dim / BROADCAST_GROUP_SIZE;
#else
                    const uint heads_dim = num_heads_dim;
#endif
                    const uint value_seq_offset = subsequence_begins[gws_seq_indexes_correspondence[target_seq_dim]];
                    uint value_offset = INPUT2_OFFSET +
                                        value_seq_offset * value_pitch +
                                        heads_dim * V_HEAD_SIZE +
                                        (start_partition_idx + seq_len_leftovers_start) * value_pitch + head_size_idx;

#else // !IS_PAGED_ATTENTION
#ifdef BEAM_TABLE_TYPE
                    const uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len_leftovers_start + sglid, sgid * SUBGROUP_SIZE)];
                    const uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1_idx, 0, 0, start_partition_idx + seq_len_leftovers_start + sglid, sgid * SUBGROUP_SIZE);
#else
                    const uint b_idx = b0_idx;
    #ifdef INPUT2_DIMS_ORDER
                    uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b0_idx, b1_idx, 0, 0, start_partition_idx + seq_len_leftovers_start, head_size_idx);
    #else
                    uint value_offset = INPUT2_GET_INDEX(b0_idx, b1_idx, start_partition_idx + seq_len_leftovers_start, head_size_idx);
    #endif
#endif
#endif

#if IS_KV_COMPRESSED
                    const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + min(seq_len_leftovers_start + sglid, seq_len_end - 1), 0);
                    // const uint comp_offset = GET_COMPRESSION_INDEX(VALUE_COMPRESSION_SCALE, b_idx, b1_idx / BROADCAST_GROUP_SIZE, start_partition_idx + seq_len_leftovers_start + sglid, 0);
                    VALUE_COMPRESSION_SCALE_TYPE comp_scale = val_scale[comp_offset];
#if USE_ASYMMETRIC_QUANTIZATION
                    VALUE_COMPRESSION_SCALE_TYPE comp_zp = val_scale[comp_offset + 1];
#endif
#endif

                    for (uint seq_len_idx = 0; seq_len_idx < partition_seq_len - seq_len_leftovers_start; seq_len_idx++) {
                    #ifdef V_HEAD_SIZE_LEFTOVER
                        #ifdef BEAM_TABLE_TYPE
                        const uint value_offset_seq = sub_group_broadcast(value_offset, seq_len_idx);
                        const INPUT2_TYPE value_packed = (head_size_idx <= V_HEAD_SIZE) ? value_input[value_offset_seq] : INPUT2_VAL_ZERO;
                        #else // !BEAM_TABLE_TYPE
                        INPUT2_TYPE value_packed;
                        if (sgid < SUBGROUPS_PER_WG - 1)
                            value_packed = VALUE_BLOCK_READ(value_input, value_offset);
                        else
                            value_packed = (sglid < V_HEAD_SIZE_LEFTOVER) ? value_input[value_offset] : INPUT2_VAL_ZERO;
                        #endif // BEAM_TABLE_TYPE

                    #else // !V_HEAD_SIZE_LEFTOVER
                        #ifdef BEAM_TABLE_TYPE
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, sub_group_broadcast(value_offset, seq_len_idx));
                        #else
                        const INPUT2_TYPE value_packed = VALUE_BLOCK_READ(value_input, value_offset);
                        #endif
                    #endif // V_HEAD_SIZE_LEFTOVER

#if IS_KV_COMPRESSED && USE_ASYMMETRIC_QUANTIZATION
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed - sub_group_broadcast(comp_zp, seq_len_idx)) * sub_group_broadcast(comp_scale, seq_len_idx);
#elif IS_KV_COMPRESSED
                        VALUE_COMPRESSION_SCALE_TYPE value_val = (value_packed * sub_group_broadcast(comp_scale, seq_len_idx));
#else
                        INPUT2_TYPE value_val = value_packed;
#endif

                        for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
                            acc_output_res[seq_idx] = mad(sub_group_broadcast(qk_val[seq_idx], seq_len_idx), value_val, acc_output_res[seq_idx]);
                        }

#ifndef BEAM_TABLE_TYPE
                        value_offset += value_pitch;
#endif
                    }
                }
            }

            // Rescale acc_output_res values and save current iter results to global accumulator
#if IS_FLASHATTEN_V2
            {
                // protect slm_qk_vals as it is read in w*v stage and write in next round q*k stage.
                barrier(CLK_LOCAL_MEM_FENCE);

                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    if (start_partition_idx > 0) {
                        OUTPUT_TYPE updated_prev_res = TO_SOFTMAX_ACCUMULATOR_TYPE(output_acc[seq_idx]) * slm_update_factor[seq_idx];
                        acc_output_res[seq_idx] += updated_prev_res;
                    }
                    output_acc[seq_idx] = acc_output_res[seq_idx];
                }
            }
#else /*!IS_FLASHATTEN_V2*/
            {
                SOFTMAX_ACCUMULATOR_TYPE exp_sum_prev = slm_exp_sum_prev[sglid];
                SOFTMAX_ACCUMULATOR_TYPE exp_sum_cur = slm_exp_sum_cur[sglid];
                SOFTMAX_ACCUMULATOR_TYPE max_val_prev = slm_max_val_prev[sglid];
                SOFTMAX_ACCUMULATOR_TYPE max_val_cur = slm_max_val_cur[sglid];

                barrier(CLK_LOCAL_MEM_FENCE);

                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
                    SOFTMAX_ACCUMULATOR_TYPE total_max = SOFTMAX_ACCUMULATOR_MAX_FUNC(sub_group_broadcast(max_val_prev, seq_idx), sub_group_broadcast(max_val_cur, seq_idx));
                    SOFTMAX_ACCUMULATOR_TYPE updated_exp_sum_prev = sub_group_broadcast(exp_sum_prev, seq_idx) * native_exp(sub_group_broadcast(max_val_prev, seq_idx) - total_max);
                    SOFTMAX_ACCUMULATOR_TYPE updated_exp_sum_cur = sub_group_broadcast(exp_sum_cur, seq_idx) * native_exp(sub_group_broadcast(max_val_cur, seq_idx) - total_max);
                    SOFTMAX_ACCUMULATOR_TYPE updated_total_exp_sum = updated_exp_sum_prev + updated_exp_sum_cur;

                    if (start_partition_idx > 0) {
                        OUTPUT_TYPE updated_prev_res = TO_SOFTMAX_ACCUMULATOR_TYPE(output_acc[seq_idx]) * updated_exp_sum_prev / updated_total_exp_sum;;
                        acc_output_res[seq_idx] *= updated_exp_sum_cur / updated_total_exp_sum;
                        acc_output_res[seq_idx] += updated_prev_res;
                    }

                    output_acc[seq_idx] = acc_output_res[seq_idx];

                    if (sgid == 0 && sglid == 0) {
                        slm_exp_sum_prev[seq_idx] = updated_total_exp_sum;
                        slm_max_val_prev[seq_idx] = total_max;
                    }
                }
            }
#endif /*!IS_FLASHATTEN_V2*/
        } /* end of QK*V calculation */
    } /* end of iter over source sequence length */

    // Combine results from multiple SGs and store to output buffer

    if (sgid >= (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) {
        unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
            slm_qk_vals[seq_idx][(uint)get_local_id(2)] = output_acc[seq_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (sgid < (SUBGROUPS_PER_WG / SG_SCALE_FACTOR)) {
        unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
            unroll_for (uint i = 1; i < SG_SCALE_FACTOR; i++) {
                output_acc[seq_idx] += slm_qk_vals[seq_idx][(i * V_HEAD_SIZE) + head_size_idx];
            }
        }

#if IS_PAGED_ATTENTION
        uint output_offset = block_start_pos * V_HEAD_SIZE * NUM_HEADS + num_heads_dim * V_HEAD_SIZE + sgid * SUBGROUP_SIZE;
        const uint output_pitch = V_HEAD_SIZE * NUM_HEADS;
#else
        uint output_offset = OUTPUT_GET_INDEX(b0_idx, b1_idx, target_seq_idx, sgid * SUBGROUP_SIZE);
        const uint output_pitch = V_HEAD_SIZE;
#endif

        #ifdef V_HEAD_SIZE_LEFTOVER
        if (TARGET_SEQ_LEN_BLOCK_SIZE > seq_idx_end) {
            if (sgid < SUBGROUPS_PER_WG - 1) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
#if IS_FLASHATTEN_V2
                    output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                    OUTPUT_BLOCK_WRITE(output, output_offset, output_acc[seq_idx]);
                    output_offset += output_pitch;
                }
            } else if (sglid < V_HEAD_SIZE_LEFTOVER) {
                for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
#if IS_FLASHATTEN_V2
                    output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                    output[output_offset + sglid] = output_acc[seq_idx];
                    output_offset += output_pitch;
                }
            }
        } else {
            if (sgid < SUBGROUPS_PER_WG - 1) {
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
#if IS_FLASHATTEN_V2
                    output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                    OUTPUT_BLOCK_WRITE(output, output_offset, output_acc[seq_idx]);
                    output_offset += output_pitch;
                }
            } else if (sglid < V_HEAD_SIZE_LEFTOVER) {
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
#if IS_FLASHATTEN_V2
                    output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                    output[output_offset + sglid] = output_acc[seq_idx];
                    output_offset += output_pitch;
                }
            }
        }
        #else // !defined(V_HEAD_SIZE_LEFTOVER)
        if (TARGET_SEQ_LEN_BLOCK_SIZE > seq_idx_end) {
            for (uint seq_idx = 0; seq_idx < seq_idx_end; seq_idx++) {
#if IS_FLASHATTEN_V2
                output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                OUTPUT_BLOCK_WRITE(output, output_offset, output_acc[seq_idx]);
                output_offset += output_pitch;
            }
        } else {
                unroll_for (uint seq_idx = 0; seq_idx < TARGET_SEQ_LEN_BLOCK_SIZE; seq_idx++) {
#if IS_FLASHATTEN_V2
                    output_acc[seq_idx] /= slm_exp_sum_prev[seq_idx];
#endif
                    OUTPUT_BLOCK_WRITE(output, output_offset, output_acc[seq_idx]);
                    output_offset += output_pitch;
                }
        }
        #endif // V_HEAD_SIZE_LEFTOVER
    }
}

#endif // TARGET_SEQ_LEN_BLOCK_SIZE != 1

#endif  // SDPA_STAGE_0

#ifdef SDPA_STAGE_1

// MTL iGPU faces high register pressure issue with a higher number of REG_VERSION_MAX_VALUES_PER_WI.
// To mitigate this, add an additional level of SDPA results processing
// with lower register pressure (REG_VERSION_MAX_VALUES_PER_WI_LOWER).

#if SOFTMAX_ACCUMULATOR_TYPE_SIZE == 4
#define REG_VERSION_MAX_VALUES_PER_WI 24
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 8
#elif SOFTMAX_ACCUMULATOR_TYPE_SIZE == 2
#define REG_VERSION_MAX_VALUES_PER_WI 48
#define REG_VERSION_MAX_VALUES_PER_WI_LOWER 16
#else
#error Unexpected SOFTMAX_ACCUMULATOR data type size
#endif

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// exp_sums      [batch, heads_num, q_len, partition_idx]
// max_logits    [batch, heads_num, q_len, partition_idx]
// tmp_out       [batch, heads_num, q_len, partition_idx, head_size]

#define MAX_PARTITIONS_NUM 128

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sdpa_opt_finalization_stage)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output,
    const __global SOFTMAX_ACCUMULATOR_TYPE* exp_sums,
    const __global SOFTMAX_ACCUMULATOR_TYPE* max_logits,
    const __global OUTPUT_TYPE* tmp_out,
    const uint num_of_partitions) {
    const uint batch_idx = get_global_id(0);
    const uint b0_idx = batch_idx / NUM_HEADS;
    const uint b1_idx = batch_idx % NUM_HEADS;
    const uint target_seq_idx = get_global_id(1);
    const uint local_id = get_local_id(2);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions) +
                                         b1_idx * (TARGET_SEQ_LEN * num_of_partitions) +
                                         target_seq_idx * (num_of_partitions);
    __global SOFTMAX_ACCUMULATOR_TYPE* cur_exp_sums = exp_sums + offset;
    __global SOFTMAX_ACCUMULATOR_TYPE* cur_max_logits = max_logits + offset;
    __local SOFTMAX_ACCUMULATOR_TYPE tmp_slm[SUBGROUP_SIZE];
    __local SOFTMAX_ACCUMULATOR_TYPE max_logits_u_exp_sum[MAX_PARTITIONS_NUM];

    SOFTMAX_ACCUMULATOR_TYPE local_max_logit = SOFTMAX_ACCUMULATOR_VAL_MIN;
    const uint reduce_offset = V_HEAD_SIZE / SUBGROUP_SIZE > SUBGROUP_SIZE ? SUBGROUP_SIZE * SUBGROUP_SIZE : V_HEAD_SIZE;
    for (uint i = local_id; i < num_of_partitions; i+= reduce_offset) {
        max_logits_u_exp_sum[i] = cur_max_logits[i];
        local_max_logit = SOFTMAX_ACCUMULATOR_MAX_FUNC(local_max_logit, max_logits_u_exp_sum[i]);
    }
    local_max_logit = sub_group_reduce_max(local_max_logit);
    if (sglid == 0) {
        tmp_slm[sgid] = local_max_logit;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (sglid < V_HEAD_SIZE / SUBGROUP_SIZE) {
        local_max_logit = tmp_slm[sglid];
    }
    SOFTMAX_ACCUMULATOR_TYPE global_max = sub_group_reduce_max(local_max_logit);

    // Update exp_sum with respect to the global maximum
    SOFTMAX_ACCUMULATOR_TYPE local_exp_sum = SOFTMAX_ACCUMULATOR_VAL_ZERO;
    for (uint i = local_id; i < num_of_partitions; i+= reduce_offset) {
         SOFTMAX_ACCUMULATOR_TYPE exp_sum_new = cur_exp_sums[i] * native_exp(max_logits_u_exp_sum[i] - global_max);
         max_logits_u_exp_sum[i] = exp_sum_new;
         local_exp_sum += exp_sum_new;
    }
    local_exp_sum = sub_group_reduce_add(local_exp_sum);
    if (sglid == 0) {
        tmp_slm[sgid] = local_exp_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    local_exp_sum = 0;
    if (sglid < V_HEAD_SIZE / SUBGROUP_SIZE) {
        local_exp_sum = tmp_slm[sglid];
    }

    SOFTMAX_ACCUMULATOR_TYPE global_exp_sum = sub_group_reduce_add(local_exp_sum);
    SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;
    for (uint partition_idx = 0; partition_idx < num_of_partitions; partition_idx++) {
            const uint tmp_out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * num_of_partitions * V_HEAD_SIZE) +
                                        b1_idx * (TARGET_SEQ_LEN * num_of_partitions * V_HEAD_SIZE) +
                                        target_seq_idx * (num_of_partitions * V_HEAD_SIZE) +
                                        partition_idx * (V_HEAD_SIZE) + local_id;
            OUTPUT_TYPE out_val = tmp_out[tmp_out_offset];
            acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * TO_SOFTMAX_ACCUMULATOR_TYPE(max_logits_u_exp_sum[partition_idx]);
    }
    const uint out_offset = b0_idx * (NUM_HEADS * TARGET_SEQ_LEN * V_HEAD_SIZE) +
                            b1_idx * (TARGET_SEQ_LEN * V_HEAD_SIZE) +
                            target_seq_idx * (V_HEAD_SIZE) +
                            local_id;

    output[out_offset] = TO_OUTPUT_TYPE(acc) / TO_OUTPUT_TYPE(global_exp_sum);
}

#endif
