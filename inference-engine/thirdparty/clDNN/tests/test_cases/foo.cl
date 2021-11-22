static std::vector<std::string> kernel_codes = {
    (std::string) R"foo(
//====================================================
// Kernel template: reorder_weights
// Kernel name: reorder_weights_1_273_weight_0
#define KERNEL(name) __kernel void reorder_weights_1_273_weight_0
#define FUNC(name)  _##name##_reorder_weights_1_273_weight_0
#define FUNC_CALL(name)  _##name##_reorder_weights_1_273_weight_0
#define FP16_SUPPORTED 0
#define FP16_UNIT_USED 0
#define INPUT0_SIZE_X 3
#define INPUT0_SIZE_Y 3
#define INPUT0_SIZE_Z 1
#define INPUT0_IFM_NUM 128
#define INPUT0_OFM_NUM 256
#define INPUT0_GROUPS_NUM 1
#define INPUT0_X_PITCH 1
#define INPUT0_Y_PITCH 3
#define INPUT0_Z_PITCH 1
#define INPUT0_IFM_PITCH 9
#define INPUT0_OFM_PITCH 1152
#define INPUT0_GROUPS_PITCH 1
#define INPUT0_OFFSET 0
#define INPUT0_VIEW_OFFSET 0
#define INPUT0_LENGTH 294912
#define INPUT0_DIMS 4
#define INPUT0_SIMPLE 1
#define INPUT0_GROUPED 0
#define INPUT0_LAYOUT_OIYX 1
#define INPUT0_TYPE char
#define INPUT0_VAL_MAX CHAR_MAX
#define INPUT0_VAL_MIN CHAR_MIN
#define INPUT0_VAL_ONE (char) 1
#define INPUT0_VAL_ZERO (char) 0
#define TO_INPUT0_TYPE(v) convert_char(v)
#define TO_INPUT0_TYPE_SAT(v) convert_char_sat(v)
#define AS_INPUT0_TYPE(v) as_char(v)
#define INPUT0_MAX_FUNC max
#define INPUT0_MIN_FUNC min
#define INPUT0_ABS_FUNC abs
#define INPUT0_TYPE_SIZE 1
#define INPUT0_IS_FP 0
#define INPUT0_SIZE 4
#define INPUT0_SIZES (size_t []){ 3,3,128,256,1,1,1,1,1, }
#define INPUT0_PITCHES (size_t []){ 1,3,9,1152,1,1,1,1,1, }
#define INPUT0_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, }
#define INPUT0_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, }
#define INPUT0_INDEX_FUNC inline uint FUNC(OIYX)(){return 0;}
#define INIT_INPUT0_INDEX_FUNC_HERE INPUT0_INDEX_FUNC
#define GET_INPUT0_OIYX_INDEX(prefix, g, o, i, z, y, x)  \
    CAT(prefix, _OFFSET) + \
    (x)*CAT(prefix, _X_PITCH) + \
    (y)*CAT(prefix, _Y_PITCH) + \
    (z)*CAT(prefix, _Z_PITCH) + \
    (i)*CAT(prefix, _IFM_PITCH) + \
    (o)*CAT(prefix, _OFM_PITCH) + \
    (g)*CAT(prefix, _GROUPS_PITCH)

#define INPUT0_GET_INDEX(o, i, y, x) GET_INPUT0_OIYX_INDEX(INPUT0, 0, o, i, 0, y, x)
#define OUTPUT_SIZE_X 3
#define OUTPUT_SIZE_Y 3
#define OUTPUT_SIZE_Z 1
#define OUTPUT_IFM_NUM 128
#define OUTPUT_OFM_NUM 256
#define OUTPUT_GROUPS_NUM 1
#define OUTPUT_X_PITCH 1
#define OUTPUT_Y_PITCH 3
#define OUTPUT_Z_PITCH 9
#define OUTPUT_IFM_PITCH 9
#define OUTPUT_OFM_PITCH 1152
#define OUTPUT_GROUPS_PITCH 1
#define OUTPUT_OFFSET 0
#define OUTPUT_VIEW_OFFSET 0
#define OUTPUT_LENGTH 294912
#define OUTPUT_DIMS 5
#define OUTPUT_SIMPLE 0
#define OUTPUT_GROUPED 0
#define OUTPUT_LAYOUT_OS_IS_ZYX_OSV16_ISV16 1
#define OUTPUT_TYPE char
#define OUTPUT_VAL_MAX CHAR_MAX
#define OUTPUT_VAL_MIN CHAR_MIN
#define OUTPUT_VAL_ONE (char) 1
#define OUTPUT_VAL_ZERO (char) 0
#define TO_OUTPUT_TYPE(v) convert_char(v)
#define TO_OUTPUT_TYPE_SAT(v) convert_char_sat(v)
#define AS_OUTPUT_TYPE(v) as_char(v)
#define OUTPUT_MAX_FUNC max
#define OUTPUT_MIN_FUNC min
#define OUTPUT_ABS_FUNC abs
#define OUTPUT_TYPE_SIZE 1
#define OUTPUT_IS_FP 0
#define OUTPUT_SIZE 5
#define OUTPUT_SIZES (size_t []){ 3,3,1,128,256,1,1,1,1, }
#define OUTPUT_PITCHES (size_t []){ 1,3,9,9,1152,1,1,1,1, }
#define OUTPUT_PAD_BEFORE (size_t []){ 0,0,0,0,0,0,0,0,0, }
#define OUTPUT_PAD_AFTER (size_t []){ 0,0,0,0,0,0,0,0,0, }
#define UNIT_TYPE float
#define UNIT_VAL_MAX FLT_MAX
#define UNIT_VAL_MIN -UNIT_VAL_MAX
#define UNIT_VAL_ONE 1.0f
#define UNIT_VAL_ZERO 0.0f
#define TO_UNIT_TYPE(v) convert_float(v)
#define TO_UNIT_TYPE_SAT(v) convert_float(v)
#define AS_UNIT_TYPE(v) as_float(v)
#define UNIT_MAX_FUNC fmax
#define UNIT_MIN_FUNC fmin
#define UNIT_ABS_FUNC fabs
#define UNIT_TYPE_SIZE 4
#define UNIT_IS_FP 1
#define SUB_GROUP_SIZE 1
)foo", (std::string) R"foo(
#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define __CAT_FUNC(x, y) FUNC(x##y)
#define CAT_FUNC(x, y) __CAT_FUNC(x, y)
#define __CAT_FUNC_CALL(x, y) FUNC_CALL(x##y)
#define CAT_FUNC_CALL(x, y) __CAT_FUNC_CALL(x, y)
#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))
#if OPT_HINTS_SUPPORTED
# define ASSUME_HINT(x) __builtin_assume(x)
#else
# define ASSUME_HINT(x) do { } while (0)
#endif
#define GET_DATA_INDEX(prefix, b, f, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_RAW(prefix, i0, i1, i2, i3) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3]
#define GET_DATA_INDEX_SAFE(prefix, b, f, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
 #define GET_DATA_INDEX_5D(prefix, b, f, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_RAW_5D(prefix, i0, i1, i2, i3, i4) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4]
#define GET_DATA_INDEX_5D_SAFE(prefix, b, f, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_6D(prefix, b, f, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (w)*CAT(prefix, _W_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 (b)*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_6D_SAFE(prefix, b, f, w, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (w % CAT(prefix, _SIZE_W ))*CAT(prefix, _W_PITCH) + \
 (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + \
 (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
#define GET_DATA_INDEX_RAW_6D(prefix, i0, i1, i2, i3, i4, i5) \
 CAT(prefix, _OFFSET) + \
 (i0)*CAT(prefix, _PITCHES)[0] + \
 (i1)*CAT(prefix, _PITCHES)[1] + \
 (i2)*CAT(prefix, _PITCHES)[2] + \
 (i3)*CAT(prefix, _PITCHES)[3] + \
 (i4)*CAT(prefix, _PITCHES)[4] + \
 (i5)*CAT(prefix, _PITCHES)[5]
#define GET_DATA_BS_FYX_BSV8_INDEX(prefix, b, f, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((b) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (f)*CAT(prefix, _FEATURE_PITCH) + \
 ((b) / (sub_group_size))*CAT(prefix, _BATCH_PITCH) \
 )
)foo", (std::string) R"foo(
inline uint FUNC(get_b_fs_yx_fsv_index)(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + b) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint FUNC(get_b_fs_yx_fsv_index_safe)(uint b, uint f, uint y, uint x,
 uint x_size, uint y_size, uint f_size, uint b_size,
 uint b_pad_before, uint b_pad_after,
 uint f_pad_before, uint f_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = (b_pad_before + (b % b_size)) * b_pitch +
 fs * fs_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
#define GET_DATA_B_FS_YX_FSV16_INDEX(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_YX_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index_safe)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_YX_FSV4_INDEX(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
#define GET_DATA_B_FS_YX_FSV4_INDEX_SAFE(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index_safe)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 4)
)foo", (std::string) R"foo(
#define GET_DATA_B_FS_YX_FSV32_INDEX(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
#define GET_DATA_B_FS_YX_FSV32_INDEX_SAFE(prefix, b, f, y, x) \
 FUNC_CALL(get_b_fs_yx_fsv_index_safe)( \
 b, f, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_BATCH_NUM), \
 CAT(prefix, _PAD_AFTER_BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
#define GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(prefix, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)( \
 0, o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
#define GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, o, i, z, y, x, sub_group_size) \
 FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)( \
 0, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
)foo", (std::string) R"foo(
inline uint FUNC(get_os_is_zyx_osv_isv_index)(uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch;
 return output_offset;
}
inline uint FUNC(get_g_os_is_zyx_osv_isv_index)(uint g, uint o, uint i, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint i_size, uint o_size, uint osv_size, uint isv_size)
{
 const uint isv = i % isv_size;
 const uint osv = o % osv_size;
 const uint is = i / isv_size;
 const uint os = o / osv_size;
 const uint x_pitch = osv_size * isv_size;
 const uint y_pitch = x_pitch * x_size;
 const uint z_pitch = y_pitch * y_size;
 const uint is_pitch = z_pitch * z_size;
 const uint os_pitch = is_pitch * ((i_size + isv_size - 1) / isv_size);
 const uint g_pitch = os_pitch * ((o_size + osv_size - 1) / osv_size);
 const uint output_offset =
 isv +
 osv * isv_size +
 x * x_pitch +
 y * y_pitch +
 z * z_pitch +
 is * is_pitch +
 os * os_pitch +
 g * g_pitch;
 return output_offset;
}
#define GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, g, o, i, z, y, x) \
 FUNC_CALL(get_g_os_is_zyx_osv_isv_index)( \
 g, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv_index)( \
 o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 1, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 16, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 32, \
 16)
#define GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 64, \
 16)
#define GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)( \
 g, o, i, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
)foo", (std::string) R"foo(
#define GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 FUNC_CALL(get_os_is_zyx_isv8_osv16_isv2_index)( \
 g, o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _GROUPS_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFFSET) \
 )
inline uint FUNC(get_os_is_zyx_isv8_osv16_isv2_index)(uint g, uint o, uint i, uint z, uint y, uint x, uint x_size, uint y_size, uint z_size,
 uint g_size, uint o_size, uint i_size, uint offset)
{
 const uint group_offset = g * o_size * i_size * z_size * y_size * x_size;
 const uint xyz_offset = (x + y * x_size + z * x_size * y_size)* 8*16*2;
 const uint i2_val = i % 2;
 const uint i2_slice = i / 2;
 const uint i8_v = i2_slice % 8;
 const uint i8_s = i2_slice / 8;
 const uint i2_offset = i2_val;
 const uint o_offset = (o % 16)*2 + (o / 16) * 16 * i_size * x_size * y_size * z_size;
 const uint i8_offset = 8*16*2* x_size*y_size*z_size * i8_s + 16*2*i8_v;
 const size_t idx = offset + group_offset + xyz_offset + i2_offset + i8_offset + o_offset;
 return idx;
}
inline uint FUNC(get_os_zyxi_osv16_index)(uint o, uint i, uint z, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size, uint z_size)
{
 const size_t idx = o%16 + (o / 16)*i_size*x_size*y_size*z_size*16 +
 16 *(i+ x*i_size + y*i_size*x_size + z*i_size*x_size*y_size);
 return idx;
}
#define GET_FILTER_OS_ZYXI_OSV16(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_zyxi_osv16_index)( \
 o, i, z, y, x, CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z))
#define GET_FILTER_GOIYX(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOYX(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOYX_SAFE(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_INDEX(prefix, g, o, i, y, x) GET_FILTER_GOIYX(prefix, g, o, i, y, x)
#define GET_FILTER_INDEX_SAFE(prefix, g, o, i, y, x) GET_FILTER_GOIYX_SAFE(prefix, g, o, i, y, x)
#define GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_GIOZYX(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 (g)*CAT(prefix, _GROUPS_PITCH)
)foo", (std::string) R"foo(
#define GET_FILTER_GIOZYX_SAFE(prefix, g, o, i, z, y, x) \
 CAT(prefix, _OFFSET) + \
 (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + \
 (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + \
 (z % CAT(prefix, _SIZE_Z ))*CAT(prefix, _Z_PITCH) + \
 (i % CAT(prefix, _IFM_NUM))*CAT(prefix, _IFM_PITCH) + \
 (o % CAT(prefix, _OFM_NUM))*CAT(prefix, _OFM_PITCH) + \
 (g % CAT(prefix, _GROUPS_NUM))*CAT(prefix, _GROUPS_PITCH)
#define GET_FILTER_INDEX_5D(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX(prefix, g, o, i, z, y, x)
#define GET_FILTER_INDEX_5D_SAFE(prefix, g, o, i, z, y, x) GET_FILTER_GOIZYX_SAFE(prefix, g, o, i, z, y, x)
#define GET_FILTER_OS_IYX_OSV8_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) + \
 (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
inline uint FUNC(get_gi_yxs_os_yxsv2_osv_index)(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch, uint i_pitch,
 uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint dst_height = i*ifm_height_pitch + y*x_size + x;
 const uint base_filter_index = y*x_size + x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}
#define GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_gi_yxs_os_yxsv2_osv_index)( \
 0, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
inline uint FUNC(get_giy_xs_os_xsv2_osv_index)(uint g, uint o, uint i, uint y, uint x, uint x_size, uint g_pitch,
 uint i_pitch, uint y_pitch, uint x_pitch, uint offset, uint sub_group_size)
{
 const uint aligned_ofm_line = x_pitch;
 const uint ifm_height_pitch = (i_pitch/aligned_ofm_line);
 const uint aligned_x_line = y_pitch / x_pitch;
 const uint dst_height = i*ifm_height_pitch + y*aligned_x_line + x;
 const uint base_filter_index = x;
 const uint aligned_height = dst_height & 0xfffffffe;
 const uint base_filter_odd = (base_filter_index & 0x1);
 uint slice_id = o / sub_group_size;
 uint id_in_slice = o % sub_group_size;
 uint slice_pitch = 2*sub_group_size;
 uint offset_in_slice = (int)(sub_group_size*base_filter_odd);
 const bool last_line_in_base_filter = (x == (x_size - 1));
 if (last_line_in_base_filter && base_filter_odd == 0)
 {
 const uint element_in_slice = 32;
 slice_id = o / element_in_slice;
 id_in_slice = o % element_in_slice;
 slice_pitch = 2*element_in_slice;
 offset_in_slice = 0;
 }
 const uint in_line = (slice_pitch*slice_id + offset_in_slice + id_in_slice);
 size_t idx = offset + aligned_height*aligned_ofm_line + in_line;
 idx += g * g_pitch;
 return idx;
}
#define GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(prefix, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_giy_xs_os_xsv2_osv_index)( \
 0, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
)foo", (std::string) R"foo(
inline uint FUNC(get_os_is_yx_isa8_osv8_isv4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
        const uint isv2_idx = i % 4;
        const uint osv_idx = o % 8;
        const uint isv1_idx = (i / 4) % 8;
        const uint is_idx = i / 32;
        const uint os_idx = o / 8;
        size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
        idx += x * 4 * 8 * 8;
        idx += y * size_x * 4 * 8 * 8;
        idx += is_idx * size_y * size_x * 4 * 8 * 8;
        idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 8;
 return idx;
}
#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_yx_isa8_osv8_isv4_index)( \
 o, i, y, x, CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint FUNC(get_os_is_zyx_isa8_osv8_isv4_index)(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 8;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 8;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
 idx += x * 4 * 8 * 8;
 idx += y * size_x * 4 * 8 * 8;
 idx += z * size_y * size_x * 4 * 8 * 8;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 8;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 8;
 return idx;
}
#define GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_isa8_osv8_isv4_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint FUNC(get_os_is_yx_isa8_osv16_isv4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += is_idx * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 16;
 return idx;
}
#define GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_yx_isa8_osv16_isv4_index)( \
 o, i, y, x, CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint FUNC(get_os_is_zyx_isa8_osv16_isv4_index)(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint ifm_slices = (size_ifm + 31)/32;
 const uint isv2_idx = i % 4;
 const uint osv_idx = o % 16;
 const uint isv1_idx = (i / 4) % 8;
 const uint is_idx = i / 32;
 const uint os_idx = o / 16;
 size_t idx = offset + isv2_idx + 4 * (osv_idx + 16 * isv1_idx);
 idx += x * 4 * 8 * 16;
 idx += y * size_x * 4 * 8 * 16;
 idx += z * size_y * size_x * 4 * 8 * 16;
 idx += is_idx * size_z * size_y * size_x * 4 * 8 * 16;
 idx += os_idx * ifm_slices * size_z * size_y * size_x * 4 * 8 * 16;
 return idx;
}
#define GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_isa8_osv16_isv4_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint FUNC(get_os_is_yx_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint f_32_aligned = ((size_ifm + 31)/32) * 32;
        const uint isv2_idx = i % 4;
        const uint osv_idx = o_swizzled % 8;
        const uint isv1_idx = (i / 4) % 8;
        const uint is_idx = i / 32;
        const uint os_idx = o_swizzled / 8;
        size_t idx = offset + isv2_idx + 4 * (osv_idx + 8 * isv1_idx);
        idx += x * 4 * 8 * 8;
        idx += y * size_x * 4 * 8 * 8;
        idx += is_idx * size_y * size_x * 4 * 8 * 8;
        idx += os_idx * (f_32_aligned/32) * size_y * size_x * 4 * 8 * 8;
 return idx;
}
)foo", (std::string) R"foo(
inline uint FUNC(get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y;
 return idx;
}
inline uint FUNC(get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint z, uint y, uint x,
 uint size_x, uint size_y, uint size_z,
 uint size_ifm, uint size_ofm, uint offset)
{
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint isv_idx = i % 4;
 const uint isa_idx = (i / 4) % 8;
 const uint is_idx = (i / 32);
 const uint osv_idx = o_swizzled % 8;
 const uint osa_idx = (o_swizzled / 8) % 4;
 const uint os_idx = (o / 32);
 const uint f_32_aligned = ((size_ifm + 31)/32);
 size_t idx = offset +
 isv_idx +
 osv_idx * 4 +
 isa_idx * 8 * 4 +
 osa_idx * 8 * 32 +
 x * 32 * 32 +
 y * size_x * 32 * 32 +
 z * size_x * size_y * 32 * 32 +
 is_idx * 32 * 32 * size_x * size_y * size_z +
 os_idx * 32 * 32 * f_32_aligned * size_x * size_y * size_z;
 return idx;
}
#define GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
        FUNC_CALL(get_os_is_yx_isa8_osv8_isv4_swizzled_by_4_index)( \
 o, i, y, x, CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_yx_osa4_isa8_osv8_isv4_swizzled_by_4_index)( \
 o, i, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
#define GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_osa4_isa8_osv8_isv4_swizzled_by_4_index)( \
 o, i, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _OFFSET))
inline uint FUNC(get_is_o_yx_isv32_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
 const uint i_val = i % 32;
 const uint i_slice = i / 32;
 const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o + o_size * i_slice) ) );
 return idx;
}
#define GET_FILTER_IS_O_YX_ISV32(prefix, o, i, y, x) \
 FUNC_CALL(get_is_o_yx_isv32_index)( \
 o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y))
)foo", (std::string) R"foo(
inline uint FUNC(get_is_o32_yx_isv32_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint o_aligned_to_32 = ((o_size + 31) / 32) * 32;
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint i_aligned_to_32 = ((i_size + 31) / 32) * 32;
 const uint i_val = i % 32;
 const uint i_slice = i / 32;
 const size_t idx = i_val + 32* (x + x_size * (y + y_size * (o_swizzled + o_aligned_to_32 * i_slice) ) );
 return idx;
}
#define GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(prefix, o, i, y, x) \
 FUNC_CALL(get_is_o32_yx_isv32_swizzled_by_4_index)( \
 o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y))
inline uint FUNC(get_os_is_y_x8_osv8_isv4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint i_aligned_to_4 = ((i_size + 3) / 4) * 4;
 const uint o_aligned_to_8 = ((o_size + 7) / 8) * 8;
 const uint x_aligned_to_8 = ((x_size + 7) / 8) * 8;
 const uint i_val = i % 4;
 const uint i_slice = i / 4;
 const uint o_val = o % 8;
 const uint o_slice = o / 8;
 const size_t idx = i_val + 4 * (o_val + 8 * ( x + x_aligned_to_8 * (y + y_size * (i_slice + (i_aligned_to_4/4) * (o_slice)))));
 return idx;
}
#define GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_y_x8_osv8_isv4_index)( \
 o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y))
inline uint FUNC(get_os_is_y_x8_osv8_isv4_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint i_size, uint o_size, uint x_size, uint y_size)
{
 const uint i_aligned_to_4 = ((i_size + 3) / 4) * 4;
 const uint o_aligned_to_8 = ((o_size + 7) / 8) * 8;
 const uint x_aligned_to_8 = ((x_size + 7) / 8) * 8;
 const uint i_val = i % 4;
 const uint i_slice = i / 4;
 const uint o_swizzled = (o % 4) * 8 + ((o % 32) / 4) + (o / 32) * 32;
 const uint o_val = o_swizzled % 8;
 const uint o_slice = o_swizzled / 8;
 const size_t idx = i_val + 4 * (o_val + 8 * ( x + x_aligned_to_8 * (y + y_size * (i_slice + (i_aligned_to_4/4) * (o_slice)))));
 return idx;
}
#define GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_y_x8_osv8_isv4_swizzled_by_4_index)( \
 o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y))
#define GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(prefix, g, o, i, y, x) \
 FUNC_CALL(get_g_os_is_yx_osv16_isv4)( \
 g, o, i, y, x, \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _OFM_PITCH), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM))
inline uint FUNC(get_g_os_is_yx_osv16_isv4)(uint g, uint o, uint i, uint y, uint x,
 uint i_size,
 uint o_size,
 uint x_size,
 uint y_size,
 uint o_num,
 uint i_num)
{
 const uint otd = 16;
 uint out_depth_tile = o / otd;
 uint od = o - out_depth_tile * otd;
 uint output_slice_size = (o_num + otd - 1) / otd;
 const uint tile = 4;
 uint id_tile = i / tile;
 uint id = i - id_tile * tile;
 uint input_slice_size = (i_num + tile - 1) / tile;
 uint idx = g * output_slice_size * input_slice_size * y_size * x_size * otd * tile
 + out_depth_tile * (o_size / tile) * otd * tile
 + id_tile * i_size * otd * tile
 + y * x_size * otd * tile
 + x * otd * tile
 + od * tile
 + id;
 return idx;
}
#define GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv4)( \
 o, i, 0, y, x, \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _OFM_PITCH), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 16)
#define GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv4)( \
 o, i, 0, y, x, \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _OFM_PITCH), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 32)
)foo", (std::string) R"foo(
#define GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(prefix, o, i, z, y, x) \
 FUNC_CALL(get_os_is_zyx_osv_isv4)( \
 o, i, z, y, x, \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _OFM_PITCH), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 32)
inline uint FUNC(get_os_is_zyx_osv_isv4)(uint o, uint i, uint z, uint y, uint x,
 uint i_size,
 uint o_size,
 uint x_size,
 uint y_size,
 uint otd)
{
 uint out_depth_tile = o / otd;
 uint od = o - out_depth_tile * otd;
 const uint tile = 4;
 uint id_tile = i / tile;
 uint id = i - id_tile * tile;
 uint idx = out_depth_tile * (o_size / tile) * otd * tile
 + id_tile * i_size * otd * tile
 + z * y_size * x_size * otd * tile
 + y * x_size * otd * tile
 + x * otd * tile
 + od * tile
 + id;
 return idx;
}
#define GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_is_yx_osv32_isv4_swizzled_by_2)( \
 o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_X))
inline uint FUNC(get_os_is_yx_osv32_isv4_swizzled_by_2)(uint o, uint i, uint y, uint x,
 uint o_size,
 uint i_size,
 uint y_size,
 uint x_size)
{
 const uint osv = 32;
 const uint os = o / osv;
 const uint ofm_block = (o % osv) % 2;
 const uint ofm_in_block = (o % osv) / 2;
 const uint tile = 4;
 const uint ifm_aligned = ((i_size + tile - 1) / tile) * tile;
 const uint ifm_tile = i / tile;
 const uint id = i - ifm_tile * tile;
 uint idx = os * ifm_aligned * y_size * x_size * osv
 + ifm_tile * y_size * x_size * osv * tile
 + y * x_size * osv * tile
 + x * osv * tile
 + ofm_block * 16 * tile
 + ofm_in_block * tile
 + id;
 return idx;
}
#define GET_DATA_FS_B_YX_FSV32_INDEX(prefix, b, f, y, x) \
 FUNC_CALL(get_fs_b_yx_fsv32_index)( \
 b, f, y, x, \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM))
inline uint FUNC(get_fs_b_yx_fsv32_index)(uint b, uint f, uint y, uint x,
 uint x_pad_before, uint x_size, uint x_pad_after,
 uint y_pad_before, uint y_size, uint y_pad_after,
 uint f_pad_before,
 uint size_b)
{
 const uint feature_tile_size = 32;
 const uint x_total_size = x_pad_before + x_size + x_pad_after;
 const uint y_total_size = y_pad_before + y_size + y_pad_after;
 const uint real_x = x + x_pad_before;
 const uint real_y = y + y_pad_before;
 const uint real_f = f + f_pad_before;
 const uint x_pitch = feature_tile_size;
 const uint y_pitch = x_pitch * x_total_size;
 const uint b_pitch = y_pitch * y_total_size;
 const uint f_tile_pitch = b_pitch * size_b;
 const uint feature_tile_number = real_f / feature_tile_size;
 const uint feature_local_number = real_f % feature_tile_size;
 size_t index = 0;
 index += feature_tile_number * f_tile_pitch;
 index += b * b_pitch;
 index += real_y * y_pitch;
 index += real_x * x_pitch;
 index += feature_local_number;
 return index;
}
)foo", (std::string) R"foo(
#define GET_DATA_B_FS_ZYX_FSV16_INDEX(prefix, b, f, z, y, x) \
 FUNC_CALL(get_b_fs_zyx_fsv_index)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
 FUNC_CALL(get_b_fs_zyx_fsv_index_safe)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16)
#define GET_DATA_B_FS_ZYX_FSV32_INDEX(prefix, b, f, z, y, x) \
 FUNC_CALL(get_b_fs_zyx_fsv_index)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
#define GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE(prefix, b, f, z, y, x) \
 FUNC_CALL(get_b_fs_zyx_fsv_index_safe)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 32)
inline uint FUNC(get_b_fs_zyx_fsv_index)(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment)
{
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = b * b_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch
 + fsv;
 return output_offset;
}
inline uint FUNC(get_b_fs_zyx_fsv_index_safe)(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after,
 uint alignment) {
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignment;
 const uint fsv = f_mod % alignment;
 const uint x_pitch = alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint b_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = b * b_pitch +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
)foo", (std::string) R"foo(
inline uint FUNC(get_bs_fs_zyx_bsv_fsv_index_safe)(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size, uint b_size,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after, uint alignmentF, uint alignmentB) {
 const uint b_mod = b % b_size;
 const uint f_mod = f_pad_before + (f % f_size);
 const uint fs = f_mod / alignmentF;
 const uint fsv = f_mod % alignmentF;
 const uint bs = b_mod / alignmentB;
 const uint bsv = b_mod % alignmentB;
 const uint x_pitch = alignmentF * alignmentB;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint b_pitch = fs_pitch * ((total_f_size + alignmentF - 1) / alignmentF);
 const uint output_offset = (bs * b_pitch) + (bsv * alignmentF) +
 fs * fs_pitch +
 (z_pad_before + (z % z_size)) * z_pitch +
 (y_pad_before + (y % y_size)) * y_pitch +
 (x_pad_before + (x % x_size)) * x_pitch
 + fsv;
 return output_offset;
}
inline uint FUNC(get_bs_fs_zyx_bsv16_fsv16_index)(uint b, uint f, uint z, uint y, uint x,
 uint x_size, uint y_size, uint z_size, uint f_size,
 uint f_pad_before, uint f_pad_after,
 uint z_pad_before, uint z_pad_after,
 uint y_pad_before, uint y_pad_after,
 uint x_pad_before, uint x_pad_after) {
 const uint alignment = 16;
 const uint feature = f + f_pad_before;
 const uint fs = feature / alignment;
 const uint fsv = feature % alignment;
 const uint bs = b / alignment;
 const uint bsv = b % alignment;
 const uint bsv_pitch = alignment;
 const uint x_pitch = bsv_pitch * alignment;
 const uint y_pitch = x_pitch * (x_pad_before + x_size + x_pad_after);
 const uint z_pitch = y_pitch * (y_pad_before + y_size + y_pad_after);
 const uint fs_pitch = z_pitch * (z_pad_before + z_size + z_pad_after);
 const uint total_f_size = f_pad_before + f_size + f_pad_after;
 const uint bs_pitch = fs_pitch * ((total_f_size + alignment - 1) / alignment);
 const uint output_offset = bs * bs_pitch +
 fs * fs_pitch +
 (z_pad_before + z) * z_pitch +
 (y_pad_before + y) * y_pitch +
 (x_pad_before + x) * x_pitch +
 bsv * bsv_pitch
 + fsv;
 return output_offset;
}
#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX(prefix, b, f, y, x) \
 FUNC_CALL(get_bs_fs_zyx_bsv16_fsv16_index)( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X))
#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX(prefix, b, f, z, y, x) \
 FUNC_CALL(get_bs_fs_zyx_bsv16_fsv16_index)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X))
)foo", (std::string) R"foo(
#define GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, y, x) \
 FUNC_CALL(get_bs_fs_zyx_bsv_fsv_index_safe)( \
 b, f, 0, y, x, \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
#define GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE(prefix, b, f, z, y, x) \
 FUNC_CALL(get_bs_fs_zyx_bsv_fsv_index_safe)( \
 b, f, z, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _SIZE_Y), \
 CAT(prefix, _SIZE_Z), \
 CAT(prefix, _FEATURE_NUM), \
 CAT(prefix, _BATCH_NUM), \
 CAT(prefix, _PAD_BEFORE_FEATURE_NUM), \
 CAT(prefix, _PAD_AFTER_FEATURE_NUM), \
 CAT(prefix, _PAD_BEFORE_SIZE_Z), \
 CAT(prefix, _PAD_AFTER_SIZE_Z), \
 CAT(prefix, _PAD_BEFORE_SIZE_Y), \
 CAT(prefix, _PAD_AFTER_SIZE_Y), \
 CAT(prefix, _PAD_BEFORE_SIZE_X), \
 CAT(prefix, _PAD_AFTER_SIZE_X), 16, 16)
inline uint FUNC(get_os_is_osv32_isv32_swizzled_by_4_index)(uint o, uint i, uint y, uint x, uint size_x, uint size_y, uint size_ifm, uint size_ofm, uint offset)
{
 const uint size_ifm_a = ((size_ifm + 31)/32) * 32;
 const uint o_hi = o / 32;
 const uint o_lo = o % 32;
 const uint i_hi = i / 32;
 const uint i_lo = i % 32;
 const uint o_lo1 = o_lo % 4;
 const uint o_lo2 = (o_lo / 4) % 8;
 const uint i_lo1 = i_lo % 4;
 const uint i_lo2 = i_lo / 4;
 const uint idx_in_group = o_lo2 * 4 + o_lo1 * (32 * 8) + i_lo2 * 32 + i_lo1;
 const uint group_idx = o_hi * (size_ifm_a / 32) + i_hi;
 return group_idx * (32 * 32) + idx_in_group;
}
#define GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(prefix, o, i, y, x)\
 FUNC_CALL(get_os_is_osv32_isv32_swizzled_by_4_index)(\
 o, i, y, x, CAT(prefix, _SIZE_X ),\
 CAT(prefix, _SIZE_Y),\
 CAT(prefix, _IFM_NUM),\
 CAT(prefix, _OFM_NUM),\
 CAT(prefix, _OFFSET))
inline uint FUNC(get_os_i_yxs_osv_yxsv4_index)(uint o, uint i, uint y, uint x, uint i_size, uint size_x, uint size_y, uint osv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint os_index = o / osv;
 uint yxs_index = yx / yxsv;
 uint osv_index = o % osv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += osv_index * yxsv;
 index += yxs_index * yxsv * osv;
 index += i * osv * yx_size_aligned;
 index += os_index * osv * yx_size_aligned * i_size;
 return index;
}
#define GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(prefix, o, i, y, x) \
 FUNC_CALL(get_os_i_yxs_osv_yxsv4_index)( \
 o, i, y, x, \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 4)
#define GET_FILTER_OS_IYX_OSV32__AI32_INDEX(prefix, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_G_OS_IYX_OSV16(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_GS_OIYX_GSV16(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((g) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH) \
 )
)foo", (std::string) R"foo(
#define GET_FILTER_GS_OIZYX_GSV16(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 ((g) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*CAT(prefix, _X_PITCH) + \
 (y)*CAT(prefix, _Y_PITCH) + \
 (z)*CAT(prefix, _Z_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 (o)*CAT(prefix, _OFM_PITCH) + \
 ((g) / (sub_group_size))*CAT(prefix, _GROUPS_PITCH) \
 )
#define GET_FILTER_G_OS_IYX_OSV16_ROTATE_180(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (CAT(prefix, _SIZE_X ) - x - 1)*CAT(prefix, _X_PITCH) + \
 (CAT(prefix, _SIZE_Y ) - y - 1)*CAT(prefix, _Y_PITCH) + \
 (i)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((o) / (sub_group_size))*(sub_group_size)*CAT(prefix, _OFM_PITCH) + \
 ((i) / (sub_group_size))*CAT(prefix, _IFM_PITCH) \
 )
#define GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX(prefix, g, o, i, z, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g)*CAT(prefix, _GROUPS_PITCH) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 (z)*(sub_group_size)*CAT(prefix, _Z_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
#define GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_gi_yxs_os_yxsv2_osv_index)( \
 g, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
#define GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 FUNC_CALL(get_giy_xs_os_xsv2_osv_index)( \
 g, o, i, y, x, \
 CAT(prefix, _SIZE_X ), \
 CAT(prefix, _GROUPS_PITCH), \
 CAT(prefix, _IFM_PITCH), \
 CAT(prefix, _Y_PITCH), \
 CAT(prefix, _X_PITCH), \
 CAT(prefix, _OFFSET), \
 sub_group_size)
inline uint FUNC(get_gs_oi_yxs_gsv_yxsv4_index)(uint g, uint o, uint i, uint y, uint x, uint o_size, uint i_size, uint size_x, uint size_y, const uint gsv) {
 const uint yxsv = 4;
 uint yx = y * size_x + x;
 uint yx_size_aligned = (size_x * size_y + yxsv - 1) / yxsv * yxsv;
 uint gs_index = g / gsv;
 uint yxs_index = yx / yxsv;
 uint gsv_index = g % gsv;
 uint yxsv_index = yx % yxsv;
 uint index = 0;
 index += yxsv_index;
 index += gsv_index * yxsv;
 index += yxs_index * yxsv * gsv;
 index += o * i * gsv * yx_size_aligned;
 index += gs_index * gsv * yx_size_aligned * o_size * i_size;
 return index;
}
#define GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(prefix, g, o, i, y, x) \
 FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 4)
)foo", (std::string) R"foo(
#define GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(prefix, g, o, i, y, x) \
 FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 16)
#define GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(prefix, g, o, i, y, x) \
 FUNC_CALL(get_gs_oi_yxs_gsv_yxsv4_index)( \
 g, o, i, y, x, \
 CAT(prefix, _OFM_NUM), \
 CAT(prefix, _IFM_NUM), \
 CAT(prefix, _SIZE_X), \
 CAT(prefix, _SIZE_Y), \
 32)
#define GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX(prefix, g, o, i, y, x, sub_group_size) \
 CAT(prefix, _OFFSET) + \
 (g * CAT(prefix, _GROUPS_PITCH)) + \
 ((o) % (sub_group_size)) + \
 (sub_group_size)*( \
 (x)*(sub_group_size)*CAT(prefix, _X_PITCH) + \
 (y)*(sub_group_size)*CAT(prefix, _Y_PITCH) + \
 ((i) % (sub_group_size)) + \
 ((i) / (sub_group_size))*(sub_group_size)*CAT(prefix, _IFM_PITCH) + \
 ((o) / (sub_group_size))*CAT(prefix, _OFM_PITCH) \
 )
inline uint FUNC(get_g_os_zyx_is_osv_isv_index)(uint g, uint o, uint i, uint z, uint y, uint x,
 uint g_size, uint o_size, uint i_size, uint z_size, uint y_size, uint x_size,
 uint osv, uint isv) {
 uint is_size = (i_size + isv - 1) / isv;
 uint os_size = (o_size + osv - 1) / osv;
 uint isv_index = i % isv;
 uint osv_index = o % osv;
 uint is_index = i / isv;
 uint os_index = o / osv;
 uint isv_pitch = 1;
 uint osv_pitch = isv_pitch * isv;
 uint is_pitch = osv_pitch * osv;
 uint x_pitch = is_pitch * is_size;
 uint y_pitch = x_pitch * x_size;
 uint z_pitch = y_pitch * y_size;
 uint os_pitch = z_pitch * z_size;
 uint g_pitch = os_pitch * os_size;
 uint index = 0;
 index += isv_index * isv_pitch;
 index += osv_index * osv_pitch;
 index += is_index * is_pitch;
 index += x * x_pitch;
 index += y * y_pitch;
 index += z * z_pitch;
 index += os_index * os_pitch;
 index += g * g_pitch;
 return index;
}
#define GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, osv, isv) \
 FUNC_CALL(get_g_os_zyx_is_osv_isv_index)( \
 g, o, i, z, y, x, \
 CAT(tensor, _GROUPS_NUM), \
 CAT(tensor, _OFM_NUM), \
 CAT(tensor, _IFM_NUM), \
 CAT(tensor, _SIZE_Z), \
 CAT(tensor, _SIZE_Y), \
 CAT(tensor, _SIZE_X), \
 osv, isv)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 16, 32)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 4)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 16)
#define GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(tensor, g, o, i, z, y, x) GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX(tensor, g, o, i, z, y, x, 32, 32)
#define DECLARE_SAMPLER const sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST
#if FP16_UNIT_USED
 #define IMAGE_READ(image, coord) read_imageh((image), imageSampler, (coord))
 #define IMAGE_WRITE(image, coord, val) write_imageh((image), (coord), (val))
#else
 #define IMAGE_READ(image, coord) read_imagef((image), imageSampler, (coord))
 #define IMAGE_WRITE(image, coord, val) write_imagef((image), (coord), (val))
#endif
inline uint8 FUNC(reshape_2_to_4)(uint o, uint i, uint y, uint x, uint dst_size_y, uint dst_size_x)
{
 uint _i = i / (dst_size_y*dst_size_x);
 uint _yx = i % (dst_size_y*dst_size_x);
 uint _y = _yx / dst_size_x;
 uint _x = _yx % dst_size_x;
 return (uint8)(0, o, _i, 0, 0, _y,_x, 0);
}
inline uint8 FUNC(reshape_4_to_2)(uint o, uint i, uint y, uint x, uint src_size_y, uint src_size_x)
{
 uint _i = i*src_size_y*src_size_x + y*src_size_x + x;
 return (uint8)(0, o, _i, 0, 0, 0, 0, 0);
}
inline uint8 FUNC(reshape_4_to_5)(uint o, uint i, uint y, uint x,
 uint src_size_f, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_f = src_pitch_y * src_size_y;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}
inline uint8 FUNC(reshape_5_to_4)(uint o, uint i, uint z, uint y, uint x,
 uint src_size_f, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_f = src_pitch_z * src_size_z;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}
inline uint8 FUNC(reshape_4_to_6)(uint o, uint i, uint y, uint x,
 uint src_size_f, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_f = src_pitch_y * src_size_y;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_w = flat_idx % dst_size_w;
 flat_idx /= dst_size_w;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, dst_w, dst_z, dst_y, dst_x, 0);
}
inline uint8 FUNC(reshape_6_to_4)(uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_f = src_pitch_w * src_size_w;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, 0, dst_y, dst_x, 0);
}
inline uint8 FUNC(reshape_6_to_5)(uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_z, uint dst_size_y, uint dst_size_x)
{
 const uint src_pitch_x = 1;
 const uint src_pitch_y = src_pitch_x * src_size_x;
 const uint src_pitch_z = src_pitch_y * src_size_y;
 const uint src_pitch_w = src_pitch_z * src_size_z;
 const uint src_pitch_f = src_pitch_w * src_size_w;
 const uint src_pitch_b = src_pitch_f * src_size_f;
 uint flat_idx = x * src_pitch_x + y * src_pitch_y + z * src_pitch_z + w * src_pitch_w + i * src_pitch_f + o * src_pitch_b;
 uint dst_x = flat_idx % dst_size_x;
 flat_idx /= dst_size_x;
 uint dst_y = flat_idx % dst_size_y;
 flat_idx /= dst_size_y;
 uint dst_z = flat_idx % dst_size_z;
 flat_idx /= dst_size_z;
 uint dst_f = flat_idx % dst_size_f;
 flat_idx /= dst_size_f;
 uint dst_b = flat_idx;
 return (uint8)(0, dst_b, dst_f, 0, dst_z, dst_y, dst_x, 0);
}
)foo", (std::string) R"foo(
inline uint8 FUNC(reshape_grouped)(uint g, uint o, uint i, uint z, uint y, uint x, uint src_size_ofm, uint dst_size_ofm)
{
 const uint flat_ofm = g * src_size_ofm + o;
 const uint dst_ofm = flat_ofm % dst_size_ofm;
 const uint dst_g = flat_ofm / dst_size_ofm;
 const uint dst_ifm = i;
 const uint dst_z = z;
 const uint dst_y = y;
 const uint dst_x = x;
 return (uint8)(dst_g, dst_ofm, dst_ifm, 0, dst_z, dst_y, dst_x, 0);
}
inline uint8 FUNC(reshape_dims)(
 uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_f, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_f, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims)
{
 if (src_dims == 4 && dst_dims == 2)
 {
 return FUNC_CALL(reshape_4_to_2)(o,i,y,x,src_size_y,src_size_x);
 }
 else if (src_dims == 2 && dst_dims == 4)
 {
 return FUNC_CALL(reshape_2_to_4)(o,i,y,x,dst_size_y,dst_size_x);
 }
 else if (src_dims == 4 && dst_dims == 6)
 {
 return FUNC_CALL(reshape_4_to_6)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_w, dst_size_z, dst_size_y, dst_size_x);
 }
 else if (src_dims == 6 && dst_dims == 4)
 {
 return FUNC_CALL(reshape_6_to_4)(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
 }
 else if (src_dims == 4 && dst_dims == 5)
 {
 return FUNC_CALL(reshape_4_to_5)(o, i, y, x, src_size_f, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
 }
 else if (src_dims == 5 && dst_dims == 4)
 {
 return FUNC_CALL(reshape_5_to_4)(o, i, z, y, x, src_size_f, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_y, dst_size_x);
 }
 else if (src_dims == 6 && dst_dims == 5)
 {
 return FUNC_CALL(reshape_6_to_5)(o, i, w, z, y, x, src_size_f, src_size_w, src_size_z, src_size_y, src_size_x, dst_size_f, dst_size_z, dst_size_y, dst_size_x);
 }
 return (uint8)(0, o, i, w, z, y, x, 0);
}
inline uint8 FUNC(reshape_dims_with_groups)(
 uint g, uint o, uint i, uint w, uint z, uint y, uint x,
 uint src_size_ofm, uint src_size_ifm, uint src_size_w, uint src_size_z, uint src_size_y, uint src_size_x,
 uint dst_size_ofm, uint dst_size_ifm, uint dst_size_w, uint dst_size_z, uint dst_size_y, uint dst_size_x,
 uint src_dims, uint dst_dims, uint src_size_groups, uint dst_size_groups)
{
 if (src_dims == 5 && dst_dims == 4)
 {
 return FUNC_CALL(reshape_grouped)(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
 }
 else if (src_dims == 6 && dst_dims == 5)
 {
 return FUNC_CALL(reshape_grouped)(g, o, i, z, y, x, src_size_ofm, dst_size_ofm);
 }
 else if (src_dims == 6 && dst_dims == 4)
 {
 return FUNC_CALL(reshape_grouped)(g, o, i, 0, y, x, src_size_ofm, dst_size_ofm);
 }
 return (uint8)(g, o, i, w, z, y, x, 0);
}
#define RESHAPE_DIMS(src_prefix, dst_prefix, o, i, w, z, y, x) \
 FUNC_CALL(reshape_dims)( \
 o, i, w, z, y, x, \
 CAT(src_prefix, _FEATURE_NUM), \
 CAT(src_prefix, _SIZE_W), \
 CAT(src_prefix, _SIZE_Z), \
 CAT(src_prefix, _SIZE_Y), \
 CAT(src_prefix, _SIZE_X), \
 CAT(dst_prefix, _FEATURE_NUM), \
 CAT(dst_prefix, _SIZE_W), \
 CAT(dst_prefix, _SIZE_Z), \
 CAT(dst_prefix, _SIZE_Y), \
 CAT(dst_prefix, _SIZE_X), \
 CAT(src_prefix, _DIMS), \
 CAT(dst_prefix, _DIMS))
#define RESHAPE_WEIGHT_DIMS(src_prefix, dst_prefix, o, i, w, z, y, x) \
 FUNC_CALL(reshape_dims)( \
 o, i, w, z, y, x, \
 CAT(src_prefix, _IFM_NUM), \
 1, \
 CAT(src_prefix, _SIZE_Z), \
 CAT(src_prefix, _SIZE_Y), \
 CAT(src_prefix, _SIZE_X), \
 CAT(dst_prefix, _IFM_NUM), \
 1, \
 CAT(dst_prefix, _SIZE_Z), \
 CAT(dst_prefix, _SIZE_Y), \
 CAT(dst_prefix, _SIZE_X), \
 CAT(src_prefix, _DIMS), \
 CAT(dst_prefix, _DIMS))
#define RESHAPE_WEIGHT_DIMS_WITH_GROUPS(src_prefix, dst_prefix, g, o, i, w, z, y, x)\
 FUNC_CALL(reshape_dims_with_groups)( \
 g, o, i, w, z, y, x, \
 CAT(src_prefix, _OFM_NUM), \
 CAT(src_prefix, _IFM_NUM), \
 1, \
 CAT(src_prefix, _SIZE_Z), \
 CAT(src_prefix, _SIZE_Y), \
 CAT(src_prefix, _SIZE_X), \
 CAT(dst_prefix, _OFM_NUM), \
 CAT(dst_prefix, _IFM_NUM), \
 1, \
 CAT(dst_prefix, _SIZE_Z), \
 CAT(dst_prefix, _SIZE_Y), \
 CAT(dst_prefix, _SIZE_X), \
 CAT(src_prefix, _DIMS), \
 CAT(dst_prefix, _DIMS), \
 CAT(src_prefix, _GROUPS_NUM), \
 CAT(dst_prefix, _GROUPS_NUM))
inline uint8 FUNC(reshape_dims3d)(uint o, uint i, uint z, uint y, uint x, uint src_size_z, uint src_size_y, uint src_size_x, uint dst_size_z, uint dst_size_y, uint dst_size_x, uint src_dims, uint dst_dims)
{
 if (src_dims == 4 && dst_dims == 5)
 {
 return (uint8)(0,o,i,1,y,x,0,0);
 }
 else if (src_dims == 5 && dst_dims == 4)
 {
 uint _y = z*src_size_y + y;
 return (uint8)(0,o,i,0,_y,x,0,0);
 }
 return (uint8)(0,o,i,z,y,x,0,0);
}
void FUNC(intel_sub_group_block_write_4)( __local uint* p, uint4 data )
{
 p[ get_sub_group_local_id() ] = data.s0;
 p += 8;
 p[ get_sub_group_local_id() ] = data.s1;
 p += 8;
 p[ get_sub_group_local_id() ] = data.s2;
 p += 8;
 p[ get_sub_group_local_id() ] = data.s3;
}
uint4 FUNC(intel_sub_group_block_read_uint4)(const __local uint* p)
{
 uint4 ret;
 uint idx = get_sub_group_local_id();
 ret.s0 = p[idx]; idx += get_max_sub_group_size();
 ret.s1 = p[idx]; idx += get_max_sub_group_size();
 ret.s2 = p[idx]; idx += get_max_sub_group_size();
 ret.s3 = p[idx]; idx += get_max_sub_group_size();
 return ret;
}
uint8 FUNC(intel_sub_group_block_read_uint8)(const __local uint* p)
{
 uint8 ret;
 uint idx = get_sub_group_local_id();
 ret.s0 = p[idx]; idx += get_max_sub_group_size();
 ret.s1 = p[idx]; idx += get_max_sub_group_size();
 ret.s2 = p[idx]; idx += get_max_sub_group_size();
 ret.s3 = p[idx]; idx += get_max_sub_group_size();
 ret.s4 = p[idx]; idx += get_max_sub_group_size();
 ret.s5 = p[idx]; idx += get_max_sub_group_size();
 ret.s6 = p[idx]; idx += get_max_sub_group_size();
 ret.s7 = p[idx]; idx += get_max_sub_group_size();
 return ret;
}
inline int FUNC(mmad_4)(char4 input, char4 weight, int acc) __attribute__((overloadable))
{
 acc += (input[0] * weight[0]);
 acc += (input[1] * weight[1]);
 acc += (input[2] * weight[2]);
 acc += (input[3] * weight[3]);
 return acc;
}
inline int FUNC(mmad_4)(char4 input, uchar4 weight, int acc) __attribute__((overloadable))
{
 acc += (input[0] * weight[0]);
 acc += (input[1] * weight[1]);
 acc += (input[2] * weight[2]);
 acc += (input[3] * weight[3]);
 return acc;
}
inline int FUNC(mmad_4)(uchar4 input, char4 weight, int acc) __attribute__((overloadable))
{
 acc += (input[0] * weight[0]);
 acc += (input[1] * weight[1]);
 acc += (input[2] * weight[2]);
 acc += (input[3] * weight[3]);
 return acc;
}
inline int FUNC(mmad_4)(uchar4 input, uchar4 weight, int acc) __attribute__((overloadable))
{
 acc += (input[0] * weight[0]);
 acc += (input[1] * weight[1]);
 acc += (input[2] * weight[2]);
 acc += (input[3] * weight[3]);
 return acc;
}
)foo", (std::string) R"foo(
inline int FUNC(mmad8)(int8 A_scalars, int8 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);
 return acc;
}
inline int FUNC(mmad8)(int8 A_scalars, uint8 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
 return acc;
}
inline int FUNC(mmad8)(uint8 A_scalars, int8 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_char4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_char4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_char4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_char4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_char4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_char4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_char4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_char4(B_vectors[7]), acc);
 return acc;
}
inline int FUNC(mmad8)(uint8 A_scalars, uint8 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
 return acc;
}
inline int FUNC(mmad16)(int16 A_scalars, int16 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_char4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_char4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_char4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_char4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_char4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_char4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_char4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_char4(B_vectors[7]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[8]), as_char4(B_vectors[8]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[9]), as_char4(B_vectors[9]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[10]), as_char4(B_vectors[10]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[11]), as_char4(B_vectors[11]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[12]), as_char4(B_vectors[12]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[13]), as_char4(B_vectors[13]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[14]), as_char4(B_vectors[14]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[15]), as_char4(B_vectors[15]), acc);
 return acc;
}
inline int FUNC(mmad16)(int16 A_scalars, uint16 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[8]), as_uchar4(B_vectors[8]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[9]), as_uchar4(B_vectors[9]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[10]), as_uchar4(B_vectors[10]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[11]), as_uchar4(B_vectors[11]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[12]), as_uchar4(B_vectors[12]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[13]), as_uchar4(B_vectors[13]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[14]), as_uchar4(B_vectors[14]), acc);
 acc = FUNC_CALL(mmad_4)(as_char4(A_scalars[15]), as_uchar4(B_vectors[15]), acc);
 return acc;
}
inline int FUNC(mmad16)(uint16 A_scalars, int16 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_char4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_char4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_char4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_char4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_char4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_char4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_char4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_char4(B_vectors[7]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[8]), as_char4(B_vectors[8]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[9]), as_char4(B_vectors[9]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[10]), as_char4(B_vectors[10]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[11]), as_char4(B_vectors[11]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[12]), as_char4(B_vectors[12]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[13]), as_char4(B_vectors[13]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[14]), as_char4(B_vectors[14]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[15]), as_char4(B_vectors[15]), acc);
 return acc;
}
)foo", (std::string) R"foo(
inline int FUNC(mmad16)(uint16 A_scalars, uint16 B_vectors, int acc) __attribute__((overloadable))
{
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[0]), as_uchar4(B_vectors[0]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[1]), as_uchar4(B_vectors[1]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[2]), as_uchar4(B_vectors[2]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[3]), as_uchar4(B_vectors[3]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[4]), as_uchar4(B_vectors[4]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[5]), as_uchar4(B_vectors[5]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[6]), as_uchar4(B_vectors[6]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[7]), as_uchar4(B_vectors[7]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[8]), as_uchar4(B_vectors[8]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[9]), as_uchar4(B_vectors[9]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[10]), as_uchar4(B_vectors[10]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[11]), as_uchar4(B_vectors[11]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[12]), as_uchar4(B_vectors[12]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[13]), as_uchar4(B_vectors[13]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[14]), as_uchar4(B_vectors[14]), acc);
 acc = FUNC_CALL(mmad_4)(as_uchar4(A_scalars[15]), as_uchar4(B_vectors[15]), acc);
 return acc;
}
inline int4 FUNC(mmad4x8)(int4 A_vectors, int8 B_vectors, int4 acc) __attribute__((overloadable))
{
 int4 ret;
 for(uint i = 0; i < 4; i++)
 {
 int8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int4 FUNC(mmad4x8)(int4 A_vectors, uint8 B_vectors, int4 acc) __attribute__((overloadable))
{
 int4 ret;
 for(uint i = 0; i < 4; i++)
 {
 int8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int4 FUNC(mmad4x8)(uint4 A_vectors, int8 B_vectors, int4 acc) __attribute__((overloadable))
{
 int4 ret;
 for(uint i = 0; i < 4; i++)
 {
 uint8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int4 FUNC(mmad4x8)(uint4 A_vectors, uint8 B_vectors, int4 acc) __attribute__((overloadable))
{
 int4 ret;
 for(uint i = 0; i < 4; i++)
 {
 uint8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
)foo", (std::string) R"foo(
inline int8 FUNC(mmad8x8)(int8 A_vectors, int8 B_vectors, int8 acc) __attribute__((overloadable))
{
 int8 ret;
 for(uint i = 0; i < 8; i++)
 {
 int8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int8 FUNC(mmad8x8)(int8 A_vectors, uint8 B_vectors, int8 acc) __attribute__((overloadable))
{
 int8 ret;
 for(uint i = 0; i < 8; i++)
 {
 int8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int8 FUNC(mmad8x8)(uint8 A_vectors, int8 B_vectors, int8 acc) __attribute__((overloadable))
{
 int8 ret;
 for(uint i = 0; i < 8; i++)
 {
 uint8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int8 FUNC(mmad8x8)(uint8 A_vectors, uint8 B_vectors, int8 acc) __attribute__((overloadable))
{
 int8 ret;
 for(uint i = 0; i < 8; i++)
 {
 uint8 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 ret[i] = FUNC_CALL(mmad8)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int16 FUNC(mmad16x16)(int16 A_vectors, int16 B_vectors, int16 acc) __attribute__((overloadable))
{
 int16 ret;
 for(uint i = 0; i < 16; i++)
 {
 int16 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
 A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
 A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
 A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
 A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
 A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
 A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
 A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
 ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int16 FUNC(mmad16x16)(int16 A_vectors, uint16 B_vectors, int16 acc) __attribute__((overloadable))
{
 int16 ret;
 for(uint i = 0; i < 16; i++)
 {
 int16 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
 A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
 A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
 A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
 A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
 A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
 A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
 A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
 ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int16 FUNC(mmad16x16)(uint16 A_vectors, int16 B_vectors, int16 acc) __attribute__((overloadable))
{
 int16 ret;
 for(uint i = 0; i < 16; i++)
 {
 uint16 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
 A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
 A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
 A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
 A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
 A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
 A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
 A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
 ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline int16 FUNC(mmad16x16)(uint16 A_vectors, uint16 B_vectors, int16 acc) __attribute__((overloadable))
{
 int16 ret;
 for(uint i = 0; i < 16; i++)
 {
 uint16 A_scalars;
 A_scalars.s0 = sub_group_broadcast(A_vectors[i], 0);
 A_scalars.s1 = sub_group_broadcast(A_vectors[i], 1);
 A_scalars.s2 = sub_group_broadcast(A_vectors[i], 2);
 A_scalars.s3 = sub_group_broadcast(A_vectors[i], 3);
 A_scalars.s4 = sub_group_broadcast(A_vectors[i], 4);
 A_scalars.s5 = sub_group_broadcast(A_vectors[i], 5);
 A_scalars.s6 = sub_group_broadcast(A_vectors[i], 6);
 A_scalars.s7 = sub_group_broadcast(A_vectors[i], 7);
 A_scalars.s8 = sub_group_broadcast(A_vectors[i], 8);
 A_scalars.s9 = sub_group_broadcast(A_vectors[i], 9);
 A_scalars.sa = sub_group_broadcast(A_vectors[i], 10);
 A_scalars.sb = sub_group_broadcast(A_vectors[i], 11);
 A_scalars.sc = sub_group_broadcast(A_vectors[i], 12);
 A_scalars.sd = sub_group_broadcast(A_vectors[i], 13);
 A_scalars.se = sub_group_broadcast(A_vectors[i], 14);
 A_scalars.sf = sub_group_broadcast(A_vectors[i], 15);
 ret[i] = FUNC_CALL(mmad16)(A_scalars, B_vectors, acc[i]);
 }
 return ret;
}
inline void FUNC(sub_group_block_write_uchar16)(__global uchar* outPtr, uchar16 v)
{
#ifdef cl_intel_subgroups_char
 intel_sub_group_block_write_uc16(outPtr, v);
#else
 uint idx = get_sub_group_local_id();
 outPtr[idx] = v.s0; idx += get_max_sub_group_size();
 outPtr[idx] = v.s1; idx += get_max_sub_group_size();
 outPtr[idx] = v.s2; idx += get_max_sub_group_size();
 outPtr[idx] = v.s3; idx += get_max_sub_group_size();
 outPtr[idx] = v.s4; idx += get_max_sub_group_size();
 outPtr[idx] = v.s5; idx += get_max_sub_group_size();
 outPtr[idx] = v.s6; idx += get_max_sub_group_size();
 outPtr[idx] = v.s7; idx += get_max_sub_group_size();
 outPtr[idx] = v.s8; idx += get_max_sub_group_size();
 outPtr[idx] = v.s9; idx += get_max_sub_group_size();
 outPtr[idx] = v.sa; idx += get_max_sub_group_size();
 outPtr[idx] = v.sb; idx += get_max_sub_group_size();
 outPtr[idx] = v.sc; idx += get_max_sub_group_size();
 outPtr[idx] = v.sd; idx += get_max_sub_group_size();
 outPtr[idx] = v.se; idx += get_max_sub_group_size();
 outPtr[idx] = v.sf; idx += get_max_sub_group_size();
#endif
}
)foo", (std::string) R"foo(
inline uchar16 FUNC(sub_group_block_read_uchar16)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
 return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
 uint idx = get_sub_group_local_id();
 uchar16 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
 ret.sa = ptr[idx]; idx += get_max_sub_group_size();
 ret.sb = ptr[idx]; idx += get_max_sub_group_size();
 ret.sc = ptr[idx]; idx += get_max_sub_group_size();
 ret.sd = ptr[idx]; idx += get_max_sub_group_size();
 ret.se = ptr[idx]; idx += get_max_sub_group_size();
 ret.sf = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline uchar16 FUNC(sub_group_block_read_uchar16)(const __local uchar* ptr) __attribute__((overloadable))
{
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
 return (uchar16)(intel_sub_group_block_read_uc8(ptr), intel_sub_group_block_read_uc8(ptr + 8 * get_max_sub_group_size()));
#else
 uint idx = get_sub_group_local_id();
 uchar16 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s8 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s9 = ptr[idx]; idx += get_max_sub_group_size();
 ret.sa = ptr[idx]; idx += get_max_sub_group_size();
 ret.sb = ptr[idx]; idx += get_max_sub_group_size();
 ret.sc = ptr[idx]; idx += get_max_sub_group_size();
 ret.sd = ptr[idx]; idx += get_max_sub_group_size();
 ret.se = ptr[idx]; idx += get_max_sub_group_size();
 ret.sf = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline void FUNC(sub_group_block_write_uchar8)(__global uchar* outPtr, uchar8 v)
{
#ifdef cl_intel_subgroups_char
 intel_sub_group_block_write_uc8(outPtr, v);
#else
 uint idx = get_sub_group_local_id();
 outPtr[idx] = v.s0; idx += get_max_sub_group_size();
 outPtr[idx] = v.s1; idx += get_max_sub_group_size();
 outPtr[idx] = v.s2; idx += get_max_sub_group_size();
 outPtr[idx] = v.s3; idx += get_max_sub_group_size();
 outPtr[idx] = v.s4; idx += get_max_sub_group_size();
 outPtr[idx] = v.s5; idx += get_max_sub_group_size();
 outPtr[idx] = v.s6; idx += get_max_sub_group_size();
 outPtr[idx] = v.s7; idx += get_max_sub_group_size();
#endif
}
inline uchar8 FUNC(sub_group_block_read_uchar8)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
 return intel_sub_group_block_read_uc8(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar8 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline uchar8 FUNC(sub_group_block_read_uchar8)(const __local uchar* ptr) __attribute__((overloadable))
{
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
 return intel_sub_group_block_read_uc8(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar8 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s4 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s5 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s6 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s7 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
)foo", (std::string) R"foo(
inline void FUNC(sub_group_block_write_uchar4)(__global uchar* outPtr, uchar4 v)
{
#ifdef cl_intel_subgroups_char
 intel_sub_group_block_write_uc4(outPtr, v);
#else
 uint idx = get_sub_group_local_id();
 outPtr[idx] = v.s0; idx += get_max_sub_group_size();
 outPtr[idx] = v.s1; idx += get_max_sub_group_size();
 outPtr[idx] = v.s2; idx += get_max_sub_group_size();
 outPtr[idx] = v.s3; idx += get_max_sub_group_size();
#endif
}
inline uchar4 FUNC(sub_group_block_read_uchar4)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
 return intel_sub_group_block_read_uc4(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar4 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline uchar4 FUNC(sub_group_block_read_uchar4)(const __local uchar* ptr) __attribute__((overloadable))
{
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
 return intel_sub_group_block_read_uc4(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar4 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s2 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s3 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline void FUNC(sub_group_block_write_uchar2)(__global uchar* outPtr, uchar2 v)
{
#ifdef cl_intel_subgroups_char
 intel_sub_group_block_write_uc2(outPtr, v);
#else
 uint idx = get_sub_group_local_id();
 outPtr[idx] = v.s0; idx += get_max_sub_group_size();
 outPtr[idx] = v.s1; idx += get_max_sub_group_size();
#endif
}
inline uchar2 FUNC(sub_group_block_read_uchar2)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
 return intel_sub_group_block_read_uc2(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar2 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline uchar2 FUNC(sub_group_block_read_uchar2)(const __local uchar* ptr) __attribute__((overloadable))
{
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
 return intel_sub_group_block_read_uc2(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar2 ret;
 ret.s0 = ptr[idx]; idx += get_max_sub_group_size();
 ret.s1 = ptr[idx]; idx += get_max_sub_group_size();
 return ret;
#endif
}
inline void FUNC(sub_group_block_write_uchar)(__global uchar* outPtr, uchar v)
{
#ifdef cl_intel_subgroups_char
 intel_sub_group_block_write_uc(outPtr, v);
#else
 uint idx = get_sub_group_local_id();
 outPtr[idx] = v;
#endif
}
inline uchar FUNC(sub_group_block_read_uchar)(const __global uchar* ptr) __attribute__((overloadable))
{
#ifdef cl_intel_subgroups_char
 return intel_sub_group_block_read_uc(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar ret;
 ret = ptr[idx];
 return ret;
#endif
}
)foo", (std::string) R"foo(
inline uchar FUNC(sub_group_block_read_uchar)(const __local uchar* ptr) __attribute__((overloadable))
{
#if LOCAL_BLOCK_IO_SUPPORTED && defined(cl_intel_subgroup_local_block_io) && defined(cl_intel_subgroups_char)
 return intel_sub_group_block_read_uc(ptr);
#else
 uint idx = get_sub_group_local_id();
 uchar ret;
 ret = ptr[idx];
 return ret;
#endif
}
#define MMAD_8(A, B, C) FUNC_CALL(mmad8)(A, B, C)
#define MMAD_16(A, B, C) FUNC_CALL(mmad16)(A, B, C)
#define MMAD_4x8(A, B, C) FUNC_CALL(mmad4x8)(A, B, C)
#define MMAD_8x8(A, B, C) FUNC_CALL(mmad8x8)(A, B, C)
#define MMAD_16x16(A, B, C) FUNC_CALL(mmad16x16)(A, B, C)
#define SLM_BLOCK_WRITE_4(A, B) (FUNC_CALL(intel_sub_group_block_write_4)(A, B))
#define SLM_BLOCK_READ_4(A) (FUNC_CALL(intel_sub_group_block_read_uint4)(A))
#define SLM_BLOCK_READ_8(A) (FUNC_CALL(intel_sub_group_block_read_uint8)(A))
#define BLOCK_READ_UC_1(ptr) FUNC_CALL(sub_group_block_read_uchar)(ptr)
#define BLOCK_READ_UC_2(ptr) FUNC_CALL(sub_group_block_read_uchar2)(ptr)
#define BLOCK_READ_UC_4(ptr) FUNC_CALL(sub_group_block_read_uchar4)(ptr)
#define BLOCK_READ_UC_8(ptr) FUNC_CALL(sub_group_block_read_uchar8)(ptr)
#define BLOCK_READ_UC_16(ptr) FUNC_CALL(sub_group_block_read_uchar16)(ptr)
#define BLOCK_WRITE_UC_1(ptr, val) FUNC_CALL(sub_group_block_write_uchar)(ptr, val)
#define BLOCK_WRITE_UC_2(ptr, val) FUNC_CALL(sub_group_block_write_uchar2)(ptr, val)
#define BLOCK_WRITE_UC_4(ptr, val) FUNC_CALL(sub_group_block_write_uchar4)(ptr, val)
#define BLOCK_WRITE_UC_8(ptr, val) FUNC_CALL(sub_group_block_write_uchar8)(ptr, val)
#define BLOCK_WRITE_UC_16(ptr, val) FUNC_CALL(sub_group_block_write_uchar16)(ptr, val)
#if !defined(ACCUMULATOR_TYPE)
 #define ACCUMULATOR_TYPE float
 #define TO_ACCUMULATOR_TYPE(v) (float)(v)
 #define ACCUMULATOR_TYPE_ZERO 0.0f
#endif
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type) elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type) CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type) CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type) CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type) CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size) CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)
#define AS_TYPE(type, val) CAT(as_, type)(val)
#define TYPE_SIZE_uchar 1
#define TYPE_SIZE_char 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short 2
#define TYPE_SIZE_half 2
#define TYPE_SIZE_int 4
#define TYPE_SIZE_uint 4
#define TYPE_SIZE_float 4
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)
#define BLOCK_RW_TYPE_size1 uchar
#define BLOCK_RW_TYPE_size2 ushort
#define BLOCK_RW_TYPE_size4 uint
#define BLOCK_RW_TYPE(type_size) CAT(BLOCK_RW_TYPE_size, type_size)
#define BLOCK_READ_FUNC_size2 intel_sub_group_block_read_us
#define BLOCK_READ_FUNC_size4 intel_sub_group_block_read
#define BLOCK_READ_FUNC(type_size) CAT(BLOCK_READ_FUNC_size, type_size)
#define BLOCK_WRITE_FUNC_size2 intel_sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4 intel_sub_group_block_write
#define BLOCK_WRITE_FUNC(type_size) CAT(BLOCK_WRITE_FUNC_size, type_size)
#define BLOCK_READN_FUNC_size1(vector_size) CAT(BLOCK_READ_UC_, vector_size)
#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size) MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size2(vector_size) BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size) BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size) CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size) CAT(BLOCK_WRITE_UC_, vector_size)
#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size) MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size) BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size) CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)
#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset) \
 BLOCK_READN_FUNC(type_size, vector_size)((const addr_space BLOCK_RW_TYPE(type_size)*)(ptr) + (offset))
#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val) \
 BLOCK_WRITEN_FUNC(type_size, vector_size)( \
 (addr_space BLOCK_RW_TYPE(type_size)*)(ptr) + (offset), \
 AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_RW_TYPE(type_size), vector_size), val))
#define BLOCK_READN(type, vector_size, ptr, offset) \
 AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))
#define BLOCK_WRITEN(type, vector_size, ptr, offset, val) \
 BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val)
#define BLOCK_READN_SLM(type, vector_size, ptr, offset) \
 AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset))
#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val) \
 BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val)
#define DT_INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset) BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset) BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset) BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset) BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)
#define DT_INPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(INPUT0_TYPE, 1, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE2(ptr, offset, val) BLOCK_WRITEN(INPUT0_TYPE, 2, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE4(ptr, offset, val) BLOCK_WRITEN(INPUT0_TYPE, 4, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE8(ptr, offset, val) BLOCK_WRITEN(INPUT0_TYPE, 8, ptr, offset, val)
#define DT_INPUT_BLOCK_WRITE16(ptr, offset, val) BLOCK_WRITEN(INPUT0_TYPE, 16, ptr, offset, val)
#define DT_OUTPUT_BLOCK_READ(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 1, ptr, offset)
#define DT_OUTPUT_BLOCK_READ2(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 2, ptr, offset)
#define DT_OUTPUT_BLOCK_READ4(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 4, ptr, offset)
#define DT_OUTPUT_BLOCK_READ8(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 8, ptr, offset)
#define DT_OUTPUT_BLOCK_READ16(ptr, offset) BLOCK_READN(OUTPUT_TYPE, 16, ptr, offset)
#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)
#define DT_BIAS_BLOCK_READ(ptr, offset) BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset) BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset) BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset) BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset) BLOCK_READN(BIAS_TYPE, 16, ptr, offset)
#define DT_BIAS_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(BIAS_TYPE, 1, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE2(ptr, offset, val) BLOCK_WRITEN(BIAS_TYPE, 2, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE4(ptr, offset, val) BLOCK_WRITEN(BIAS_TYPE, 4, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE8(ptr, offset, val) BLOCK_WRITEN(BIAS_TYPE, 8, ptr, offset, val)
#define DT_BIAS_BLOCK_WRITE16(ptr, offset, val) BLOCK_WRITEN(BIAS_TYPE, 16, ptr, offset, val)
#define DT_FILTER_BLOCK_READ(ptr, offset) BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset) BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset) BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset) BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset) BLOCK_READN(FILTER_TYPE, 16, ptr, offset)
#define DT_FILTER_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(FILTER_TYPE, 1, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE2(ptr, offset, val) BLOCK_WRITEN(FILTER_TYPE, 2, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE4(ptr, offset, val) BLOCK_WRITEN(FILTER_TYPE, 4, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE8(ptr, offset, val) BLOCK_WRITEN(FILTER_TYPE, 8, ptr, offset, val)
#define DT_FILTER_BLOCK_WRITE16(ptr, offset, val) BLOCK_WRITEN(FILTER_TYPE, 16, ptr, offset, val)
)foo", (std::string) R"foo(
inline uint FUNC(get_input_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if INPUT0_SIMPLE && INPUT0_DIMS <= 4
 return GET_FILTER_INDEX(INPUT0, 0, o, i, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
 return GET_FILTER_INDEX_5D(INPUT0, 0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16 || \
 defined INPUT0_LAYOUT_OS_I_OSV16 || \
 defined INPUT0_LAYOUT_OS_I_OSV8__AI8 || \
 defined INPUT0_LAYOUT_OS_I_OSV16__AI8
 return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IYX_OSV32
 return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV32__AI32
 return GET_FILTER_OS_IYX_OSV32__AI32_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_IYX_OSV64
 return GET_FILTER_OS_IYX_OSV8_INDEX(INPUT0, o, i, y, x, 64);
#elif defined INPUT0_LAYOUT_OS_IYX_OSV16_ROTATE_180
 return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_I_YXS_OS_YXSV2_OSV16
 return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
 return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
 #error - not supported yet
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
        return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
 return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISA8_OSV16_ISV4
        return GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISA8_OSV16_ISV4
 return GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_IS_O_YX_ISV32
 return GET_FILTER_IS_O_YX_ISV32(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
 return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
 return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV16_OSV16
 return GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OIYX_O16
 return GET_FILTER_OIYX_O16(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV16_OSV16
 return GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IS_OS_ZYX_ISV16_OSV16
 return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_IS_OS_YX_ISV16_OSV16
 return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
 return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
 return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(INPUT0, o, i, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_OS_ZYXI_OSV16
 return GET_FILTER_OS_ZYXI_OSV16(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_I_YXS_OSV4_YXSV4
 return GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_GOIZYX
 return GET_FILTER_GOIZYX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GIOZYX
 return GET_FILTER_GIOZYX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_IYX_OSV16
 return GET_FILTER_G_OS_IYX_OSV16(INPUT0, g, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_G_OS_IYX_OSV32
 return GET_FILTER_G_OS_IYX_OSV16(INPUT0, g, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_GS_OIYX_GSV16
 return GET_FILTER_GS_OIYX_GSV16(INPUT0, g, o, i, y, x, 16);
#elif defined INPUT0_LAYOUT_GS_OIZYX_GSV16
 return GET_FILTER_GS_OIZYX_GSV16(INPUT0, g, o, i, z, y, x, 16);
#elif defined INPUT0_LAYOUT_GS_OIYX_GSV32
 return GET_FILTER_GS_OIYX_GSV16(INPUT0, g, o, i, y, x, 32);
#elif defined INPUT0_LAYOUT_GYXIO || \
 defined INPUT0_LAYOUT_GOIYX || \
 defined INPUT0_LAYOUT_GIOYX
 return GET_FILTER_GOIYX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_YX_OSV16_ISV16
 return GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(INPUT0, o, i, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV16_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_IS_ZYX_OSV16_ISV16
 return GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV32_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_OS_IS_ZYX_OSV64_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(INPUT0, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_GS_OI_YXS_GSV16_YXSV4
 return GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_GS_OI_YXS_GSV32_YXSV4
 return GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(INPUT0, g, o, i, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV4
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV16
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV16_ISV32
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV4
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV16
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(INPUT0, g, o, i, z, y, x);
#elif defined INPUT0_LAYOUT_G_OS_ZYX_IS_OSV32_ISV32
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(INPUT0, g, o, i, z, y, x);
#else
#error reorder_weights.cl: input format - not supported
#endif
}
)foo", (std::string) R"foo(
inline uint FUNC(get_output_index)(uint g, uint o, uint i, uint z, uint y, uint x)
{
#if OUTPUT_SIMPLE && OUTPUT_DIMS <= 4
 return GET_FILTER_INDEX(OUTPUT, 0, o, i, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
 return GET_FILTER_INDEX_5D(OUTPUT, 0, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16 || \
 defined OUTPUT_LAYOUT_OS_I_OSV16 || \
 defined OUTPUT_LAYOUT_OS_I_OSV8__AI8 || \
 defined OUTPUT_LAYOUT_OS_I_OSV16__AI8
 return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32
 return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV32__AI32
 return GET_FILTER_OS_IYX_OSV32__AI32_INDEX(OUTPUT, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV64
 return GET_FILTER_OS_IYX_OSV8_INDEX(OUTPUT, o, i, y, x, 64);
#elif defined OUTPUT_LAYOUT_OS_IYX_OSV16_ROTATE_180
 return GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_I_YXS_OS_YXSV2_OSV16
 return GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_IY_XS_OS_XSV2_OSV8__AO32
 return GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
 return 0;
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4
        return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISA8_OSV8_ISV4
 return GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV16_ISV4
        return GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISA8_OSV16_ISV4
 return GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_IS_O_YX_ISV32
 return GET_FILTER_IS_O_YX_ISV32(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_IS_O32_YX_ISV32_SWIZZLED_BY_4
 return GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4
 return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV4
 return GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2
 return GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV32_ISV4
 return GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV32_ISV4
 return GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV16_OSV16
 return GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OIYX_O16
 return GET_FILTER_OIYX_O16(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV16_OSV16
 return GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IS_OS_ZYX_ISV16_OSV16
 return GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_IS_OS_YX_ISV16_OSV16
 return GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_OSV32_ISV32_SWIZZLED_BY_4
 return GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_ISV8_OSV16_ISV2
 return GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_ISV8_OSV16_ISV2
 return GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(OUTPUT, o, i, z, y, x, SUB_GROUP_SIZE);
)foo", (std::string) R"foo(
#elif defined OUTPUT_LAYOUT_OS_ZYXI_OSV16
 return GET_FILTER_OS_ZYXI_OSV16(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_I_YXS_OSV4_YXSV4
 return GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GOIZYX || defined OUTPUT_LAYOUT_GIOZYX
 return GET_FILTER_INDEX_5D(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IYX_OSV16
 return GET_FILTER_G_OS_IYX_OSV16(OUTPUT, g, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_G_OS_IYX_OSV32
 return GET_FILTER_G_OS_IYX_OSV16(OUTPUT, g, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_GS_OIYX_GSV16
 return GET_FILTER_GS_OIYX_GSV16(OUTPUT, g, o, i, y, x, 16);
#elif defined OUTPUT_LAYOUT_GS_OIZYX_GSV16
 return GET_FILTER_GS_OIZYX_GSV16(OUTPUT, g, o, i, z, y, x, 16);
#elif defined OUTPUT_LAYOUT_GS_OIYX_GSV32
 return GET_FILTER_GS_OIYX_GSV16(OUTPUT, g, o, i, y, x, 32);
#elif defined OUTPUT_LAYOUT_GYXIO || \
 defined OUTPUT_LAYOUT_GOIYX || \
 defined OUTPUT_LAYOUT_GIOYX
 return GET_FILTER_GOIYX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GI_YXS_OS_YXSV2_OSV16
 return GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_IS_OS_ZYX_ISV16_OSV16
 return GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_IS_OS_YX_ISV16_OSV16
 return GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_ISV16_OSV16
 return GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_ISV8_OSV16_ISV2
 return GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_ISV8_OSV16_ISV2
 return GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX(OUTPUT, g, o, i, z, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_GIY_XS_OS_XSV2_OSV16__AO32 || defined OUTPUT_LAYOUT_GIY_XS_OS_XSV2_OSV8__AO32
 return GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV4_YXSV4
 return GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_ISV16_OSV16
 return GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX(OUTPUT, g, o, i, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_OS_IS_YX_OSV16_ISV16
 return GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX(OUTPUT, o, i, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV16_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_ZYX_OSV16_ISV16
 return GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV32_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_OS_IS_ZYX_OSV64_ISV16
 return GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX(OUTPUT, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV16_YXSV4
 return GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_GS_OI_YXS_GSV32_YXSV4
 return GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_IS_YX_OSV16_ISV4
 return GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX(OUTPUT, g, o, i, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV4
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV16
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV16_ISV32
 return GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV4
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV16
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX(OUTPUT, g, o, i, z, y, x);
#elif defined OUTPUT_LAYOUT_G_OS_ZYX_IS_OSV32_ISV32
 return GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX(OUTPUT, g, o, i, z, y, x);
#else
#error reorder_weights.cl: output format - not supported
#endif
}
#if OUTPUT_LAYOUT_IMAGE_2D_WEIGHTS_C1_B_FYX
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
 const unsigned o = get_global_id(0);
 const unsigned iyx = get_global_id(1);
 const unsigned x = iyx % INPUT0_SIZE_X;
 const unsigned y = (iyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
 const unsigned i = (iyx / INPUT0_SIZE_X) / INPUT0_SIZE_Y;
 MAKE_VECTOR_TYPE(UNIT_TYPE, 4) input_val = (MAKE_VECTOR_TYPE(UNIT_TYPE, 4))(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);
 const int2 coord = (int2)(o, iyx);
 uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, 0, y, x);
 input_val.s0 = TO_OUTPUT_TYPE(input[FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[4],ir[5],ir[6])]);
 IMAGE_WRITE(output, coord, input_val);
}
#else
)foo", (std::string) R"foo(
KERNEL (reorder_weights)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if OUTPUT_GROUPS_NUM > 1
 const unsigned g = (uint)get_global_id(0) / OUTPUT_OFM_NUM;
 const unsigned o = (uint)get_global_id(0) % OUTPUT_OFM_NUM;
#else
 const unsigned g = 0;
 const unsigned o = (uint)get_global_id(0);
#endif
 const unsigned i = (uint)get_global_id(1);
#if OUTPUT_DIMS == 2 || (OUTPUT_DIMS == 3 && OUTPUT_GROUPED)
 const unsigned x = 0;
 const unsigned y = 0;
 const unsigned z = 0;
#elif OUTPUT_DIMS == 4 || (OUTPUT_DIMS == 5 && OUTPUT_GROUPED)
 const unsigned x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
 const unsigned y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
 const unsigned z = 0;
#elif OUTPUT_DIMS == 5 || (OUTPUT_DIMS == 6 && OUTPUT_GROUPED)
 const unsigned zyx = get_global_id(2);
 const unsigned x = zyx % OUTPUT_SIZE_X;
 const unsigned y = (zyx / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
 const unsigned z = (zyx / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
#endif
#if OUTPUT_GROUPS_NUM > 1
 uint8 ir = RESHAPE_WEIGHT_DIMS_WITH_GROUPS(OUTPUT, INPUT0, g, o, i, 0, z, y, x);
#else
 uint8 ir = RESHAPE_WEIGHT_DIMS(OUTPUT, INPUT0, o, i, 0, z, y, x);
#endif
 uint input_idx = FUNC_CALL(get_input_index)(ir[0],ir[1],ir[2],ir[4],ir[5],ir[6]);
#if !REORDER_ROTATE
 uint output_idx = FUNC_CALL(get_output_index)(g, o, i, z, y, x);
#else
 uint output_idx = FUNC_CALL(get_output_index)(g, o, i, OUTPUT_SIZE_Z - z - 1, OUTPUT_SIZE_Y - y - 1, OUTPUT_SIZE_X - x - 1);
#endif
 output[output_idx] = TO_OUTPUT_TYPE(input[input_idx]);
}
#endif
#ifdef GET_DATA_INDEX
#undef GET_DATA_INDEX
#endif
#ifdef GET_DATA_INDEX_RAW
#undef GET_DATA_INDEX_RAW
#endif
#ifdef GET_DATA_INDEX_SAFE
#undef GET_DATA_INDEX_SAFE
#endif
#ifdef GET_DATA_INDEX_5D
#undef GET_DATA_INDEX_5D
#endif
#ifdef GET_DATA_INDEX_RAW_5D
#undef GET_DATA_INDEX_RAW_5D
#endif
#ifdef GET_DATA_INDEX_5D_SAFE
#undef GET_DATA_INDEX_5D_SAFE
#endif
#ifdef GET_DATA_INDEX_6D
#undef GET_DATA_INDEX_6D
#endif
#ifdef GET_DATA_INDEX_6D_SAFE
#undef GET_DATA_INDEX_6D_SAFE
#endif
#ifdef GET_DATA_INDEX_RAW_6D
#undef GET_DATA_INDEX_RAW_6D
#endif
#ifdef GET_DATA_BS_FYX_BSV8_INDEX
#undef GET_DATA_BS_FYX_BSV8_INDEX
#endif
#ifdef GET_DATA_B_FS_YX_FSV16_INDEX
#undef GET_DATA_B_FS_YX_FSV16_INDEX
#endif
#ifdef GET_DATA_B_FS_YX_FSV16_INDEX_SAFE
#undef GET_DATA_B_FS_YX_FSV16_INDEX_SAFE
#endif
#ifdef GET_DATA_B_FS_YX_FSV4_INDEX
#undef GET_DATA_B_FS_YX_FSV4_INDEX
#endif
#ifdef GET_DATA_B_FS_YX_FSV4_INDEX_SAFE
#undef GET_DATA_B_FS_YX_FSV4_INDEX_SAFE
#endif
#ifdef GET_DATA_B_FS_YX_FSV32_INDEX
#undef GET_DATA_B_FS_YX_FSV32_INDEX
#endif
#ifdef GET_DATA_B_FS_YX_FSV32_INDEX_SAFE
#undef GET_DATA_B_FS_YX_FSV32_INDEX_SAFE
#endif
#ifdef GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX
#undef GET_FILTER_OS_IS_YX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX
#undef GET_FILTER_OS_IS_ZYX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX
#undef GET_FILTER_IS_OS_ZYX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX
#undef GET_FILTER_IS_OS_YX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX
#undef GET_FILTER_OS_IS_YX_ISV8_OSV16_ISV2_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX
#undef GET_FILTER_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX
#endif
#ifdef GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX
#undef GET_FILTER_G_OS_IS_ZYX_OSV16_ISV16_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX
#undef GET_FILTER_OS_IS_YX_OSV16_ISV16_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX
#undef GET_FILTER_OS_IS_ZYX_OSV16_ISV16_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX
#undef GET_FILTER_OS_IS_ZYX_OSV32_ISV16_INDEX
#endif
)foo", (std::string) R"foo(
#ifdef GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX
#undef GET_FILTER_OS_IS_ZYX_OSV64_ISV16_INDEX
#endif
#ifdef GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX
#undef GET_FILTER_G_OS_IS_YX_ISV8_OSV16_ISV2_INDEX
#endif
#ifdef GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX
#undef GET_FILTER_G_OS_IS_ZYX_ISV8_OSV16_ISV2_INDEX
#endif
#ifdef GET_FILTER_OS_ZYXI_OSV16
#undef GET_FILTER_OS_ZYXI_OSV16
#endif
#ifdef GET_FILTER_GOIYX
#undef GET_FILTER_GOIYX
#endif
#ifdef GET_FILTER_GIOYX
#undef GET_FILTER_GIOYX
#endif
#ifdef GET_FILTER_GIOYX_SAFE
#undef GET_FILTER_GIOYX_SAFE
#endif
#ifdef GET_FILTER_GOIYX_SAFE
#undef GET_FILTER_GOIYX_SAFE
#endif
#ifdef GET_FILTER_INDEX
#undef GET_FILTER_INDEX
#endif
#ifdef GET_FILTER_INDEX_SAFE
#undef GET_FILTER_INDEX_SAFE
#endif
#ifdef GET_FILTER_GOIZYX
#undef GET_FILTER_GOIZYX
#endif
#ifdef GET_FILTER_GOIZYX_SAFE
#undef GET_FILTER_GOIZYX_SAFE
#endif
#ifdef GET_FILTER_GIOZYX
#undef GET_FILTER_GIOZYX
#endif
#ifdef GET_FILTER_GIOZYX_SAFE
#undef GET_FILTER_GIOZYX_SAFE
#endif
#ifdef GET_FILTER_INDEX_5D
#undef GET_FILTER_INDEX_5D
#endif
#ifdef GET_FILTER_INDEX_5D_SAFE
#undef GET_FILTER_INDEX_5D_SAFE
#endif
#ifdef GET_FILTER_OS_IYX_OSV8_INDEX
#undef GET_FILTER_OS_IYX_OSV8_INDEX
#endif
#ifdef GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX
#undef GET_FILTER_OS_IYX_OSV8_ROTATE_180_INDEX
#endif
#ifdef GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX
#undef GET_FILTER_I_YXS_OS_YXSV2_OSV_INDEX
#endif
#ifdef GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX
#undef GET_FILTER_IY_XS_OS_XSV2_OSV_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX
#undef GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX
#undef GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX
#undef GET_FILTER_OS_IS_YX_ISA8_OSV16_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX
#undef GET_FILTER_OS_IS_ZYX_ISA8_OSV16_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#undef GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#undef GET_FILTER_OS_IS_YX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#undef GET_FILTER_OS_IS_ZYX_OSA4_ISA8_OSV8_ISV4_SWIZZLED_BY_4_INDEX
#endif
#ifdef GET_FILTER_IS_O_YX_ISV32
#undef GET_FILTER_IS_O_YX_ISV32
#endif
#ifdef GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4
#undef GET_FILTER_IS_O32_YX_ISV32_SWIZZLED_BY_4
#endif
#ifdef GET_FILTER_OS_IS_Y_X8_OSV8_ISV4
#undef GET_FILTER_OS_IS_Y_X8_OSV8_ISV4
#endif
#ifdef GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
#undef GET_FILTER_OS_IS_Y_X8_OSV8_ISV4_SWIZZLED_BY_4
#endif
)foo", (std::string) R"foo(
#ifdef GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX
#undef GET_FILTER_G_OS_IS_YX_OSV16_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX
#undef GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX
#undef GET_FILTER_OS_IS_YX_OSV32_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX
#undef GET_FILTER_OS_IS_ZYX_OSV32_ISV4_INDEX
#endif
#ifdef GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX
#undef GET_FILTER_OS_IS_YX_OSV32_ISV4_SWIZZLED_BY_2_INDEX
#endif
#ifdef GET_DATA_FS_B_YX_FSV32_INDEX
#undef GET_DATA_FS_B_YX_FSV32_INDEX
#endif
#ifdef GET_DATA_B_FS_ZYX_FSV16_INDEX
#undef GET_DATA_B_FS_ZYX_FSV16_INDEX
#endif
#ifdef GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE
#undef GET_DATA_B_FS_ZYX_FSV16_INDEX_SAFE
#endif
#ifdef GET_DATA_B_FS_ZYX_FSV32_INDEX
#undef GET_DATA_B_FS_ZYX_FSV32_INDEX
#endif
#ifdef GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE
#undef GET_DATA_B_FS_ZYX_FSV32_INDEX_SAFE
#endif
#ifdef GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX
#undef GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX
#endif
#ifdef GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX
#undef GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX
#endif
#ifdef GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE
#undef GET_DATA_BS_FS_YX_BSV16_FSV16_INDEX_SAFE
#endif
#ifdef GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE
#undef GET_DATA_BS_FS_ZYX_BSV16_FSV16_INDEX_SAFE
#endif
#ifdef GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX
#undef GET_FILTER_OS_IS_OSV32_ISV32_SWIZZLED_BY_4_INDEX
#endif
#ifdef GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX
#undef GET_FILTER_OS_I_YXS_OSV4_YXSV4_INDEX
#endif
#ifdef GET_FILTER_OS_IYX_OSV32__AI32_INDEX
#undef GET_FILTER_OS_IYX_OSV32__AI32_INDEX
#endif
#ifdef GET_FILTER_G_OS_IYX_OSV16
#undef GET_FILTER_G_OS_IYX_OSV16
#endif
#ifdef GET_FILTER_GS_OIYX_GSV16
#undef GET_FILTER_GS_OIYX_GSV16
#endif
#ifdef GET_FILTER_GS_OIZYX_GSV16
#undef GET_FILTER_GS_OIZYX_GSV16
#endif
#ifdef GET_FILTER_G_OS_IYX_OSV16_ROTATE_180
#undef GET_FILTER_G_OS_IYX_OSV16_ROTATE_180
#endif
#ifdef GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX
#undef GET_FILTER_G_IS_OS_ZYX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX
#undef GET_FILTER_G_IS_OS_YX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX
#undef GET_FILTER_G_OS_IS_ZYX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX
#undef GET_FILTER_GI_YXS_OS_YXSV2_OSV_INDEX
#endif
#ifdef GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX
#undef GET_FILTER_GIY_XS_OS_XSV2_OSV_INDEX
#endif
#ifdef GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX
#undef GET_FILTER_GS_OI_YXS_GSV4_YXSV4_INDEX
#endif
#ifdef GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX
#undef GET_FILTER_GS_OI_YXS_GSV16_YXSV4_INDEX
#endif
#ifdef GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX
#undef GET_FILTER_GS_OI_YXS_GSV32_YXSV4_INDEX
#endif
#ifdef GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX
#undef GET_FILTER_G_OS_IS_YX_ISV16_OSV16_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV_ISV_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV4_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV16_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV16_ISV32_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV4_INDEX
#endif
)foo", (std::string) R"foo(
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV16_INDEX
#endif
#ifdef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX
#undef GET_FILTER_G_OS_ZYX_IS_OSV32_ISV32_INDEX
#endif
#ifdef DECLARE_SAMPLER
#undef DECLARE_SAMPLER
#endif
#ifdef IMAGE_READ
#undef IMAGE_READ
#endif
#ifdef IMAGE_WRITE
#undef IMAGE_WRITE
#endif
#ifdef IMAGE_READ
#undef IMAGE_READ
#endif
#ifdef IMAGE_WRITE
#undef IMAGE_WRITE
#endif
#ifdef __CAT
#undef __CAT
#endif
#ifdef CAT
#undef CAT
#endif
#ifdef __CAT_FUNC
#undef __CAT_FUNC
#endif
#ifdef CAT_FUNC
#undef CAT_FUNC
#endif
#ifdef __CAT_FUNC_CALL
#undef __CAT_FUNC_CALL
#endif
#ifdef CAT_FUNC_CALL
#undef CAT_FUNC_CALL
#endif
#ifdef OFFSET_GLOBAL_PTR
#undef OFFSET_GLOBAL_PTR
#endif
#ifdef MULTIPLY_OFFSET
#undef MULTIPLY_OFFSET
#endif
#ifdef RESHAPE_DIMS
#undef RESHAPE_DIMS
#endif
#ifdef RESHAPE_WEIGHT_DIMS
#undef RESHAPE_WEIGHT_DIMS
#endif
#ifdef RESHAPE_WEIGHT_DIMS_WITH_GROUPS
#undef RESHAPE_WEIGHT_DIMS_WITH_GROUPS
#endif
#ifdef ACCUMULATOR_TYPE
#undef ACCUMULATOR_TYPE
#endif
#ifdef TO_ACCUMULATOR_TYPE
#undef TO_ACCUMULATOR_TYPE
#endif
#ifdef ACCUMULATOR_TYPE_ZERO
#undef ACCUMULATOR_TYPE_ZERO
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_1
#undef MAKE_VECTOR_TYPE_IMPL_1
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_2
#undef MAKE_VECTOR_TYPE_IMPL_2
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_3
#undef MAKE_VECTOR_TYPE_IMPL_3
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_4
#undef MAKE_VECTOR_TYPE_IMPL_4
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_8
#undef MAKE_VECTOR_TYPE_IMPL_8
#endif
#ifdef MAKE_VECTOR_TYPE_IMPL_16
#undef MAKE_VECTOR_TYPE_IMPL_16
#endif
#ifdef MAKE_VECTOR_TYPE
#undef MAKE_VECTOR_TYPE
#endif
#ifdef AS_TYPE
#undef AS_TYPE
#endif
#ifdef TYPE_SIZE_uchar
#undef TYPE_SIZE_uchar
#endif
)foo", (std::string) R"foo(
#ifdef TYPE_SIZE_char
#undef TYPE_SIZE_char
#endif
#ifdef TYPE_SIZE_ushort
#undef TYPE_SIZE_ushort
#endif
#ifdef TYPE_SIZE_short
#undef TYPE_SIZE_short
#endif
#ifdef TYPE_SIZE_half
#undef TYPE_SIZE_half
#endif
#ifdef TYPE_SIZE_int
#undef TYPE_SIZE_int
#endif
#ifdef TYPE_SIZE_uint
#undef TYPE_SIZE_uint
#endif
#ifdef TYPE_SIZE_float
#undef TYPE_SIZE_float
#endif
#ifdef TYPE_SIZE
#undef TYPE_SIZE
#endif
#ifdef BLOCK_RW_TYPE_size1
#undef BLOCK_RW_TYPE_size1
#endif
#ifdef BLOCK_RW_TYPE_size2
#undef BLOCK_RW_TYPE_size2
#endif
#ifdef BLOCK_RW_TYPE_size4
#undef BLOCK_RW_TYPE_size4
#endif
#ifdef BLOCK_RW_TYPE
#undef BLOCK_RW_TYPE
#endif
#ifdef BLOCK_READ_FUNC_size2
#undef BLOCK_READ_FUNC_size2
#endif
#ifdef BLOCK_READ_FUNC_size4
#undef BLOCK_READ_FUNC_size4
#endif
#ifdef BLOCK_READ_FUNC
#undef BLOCK_READ_FUNC
#endif
#ifdef BLOCK_WRITE_FUNC_size2
#undef BLOCK_WRITE_FUNC_size2
#endif
#ifdef BLOCK_WRITE_FUNC_size4
#undef BLOCK_WRITE_FUNC_size4
#endif
#ifdef BLOCK_WRITE_FUNC
#undef BLOCK_WRITE_FUNC
#endif
#ifdef BLOCK_READN_FUNC_size1
#undef BLOCK_READN_FUNC_size1
#endif
#ifdef BLOCK_READN_FUNC_SIZE_DEF
#undef BLOCK_READN_FUNC_SIZE_DEF
#endif
#ifdef BLOCK_READN_FUNC_size2
#undef BLOCK_READN_FUNC_size2
#endif
#ifdef BLOCK_READN_FUNC_size4
#undef BLOCK_READN_FUNC_size4
#endif
#ifdef BLOCK_READN_FUNC
#undef BLOCK_READN_FUNC
#endif
#ifdef BLOCK_WRITEN_FUNC_size1
#undef BLOCK_WRITEN_FUNC_size1
#endif
#ifdef BLOCK_WRITEN_FUNC_SIZE_DEF
#undef BLOCK_WRITEN_FUNC_SIZE_DEF
#endif
#ifdef BLOCK_WRITEN_FUNC_size2
#undef BLOCK_WRITEN_FUNC_size2
#endif
#ifdef BLOCK_WRITEN_FUNC_size4
#undef BLOCK_WRITEN_FUNC_size4
#endif
#ifdef BLOCK_WRITEN_FUNC
#undef BLOCK_WRITEN_FUNC
#endif
#ifdef BLOCK_READN_RAW
#undef BLOCK_READN_RAW
#endif
#ifdef BLOCK_WRITEN_RAW
#undef BLOCK_WRITEN_RAW
#endif
#ifdef BLOCK_READN
#undef BLOCK_READN
#endif
#ifdef BLOCK_WRITEN
#undef BLOCK_WRITEN
#endif
#ifdef BLOCK_READN_SLM
#undef BLOCK_READN_SLM
#endif
#ifdef BLOCK_WRITEN_SLM
#undef BLOCK_WRITEN_SLM
#endif
#ifdef DT_INPUT_BLOCK_READ
#undef DT_INPUT_BLOCK_READ
#endif
#ifdef DT_INPUT_BLOCK_READ2
#undef DT_INPUT_BLOCK_READ2
#endif
)foo", (std::string) R"foo(
#ifdef DT_INPUT_BLOCK_READ4
#undef DT_INPUT_BLOCK_READ4
#endif
#ifdef DT_INPUT_BLOCK_READ8
#undef DT_INPUT_BLOCK_READ8
#endif
#ifdef DT_INPUT_BLOCK_READ16
#undef DT_INPUT_BLOCK_READ16
#endif
#ifdef DT_INPUT_BLOCK_WRITE
#undef DT_INPUT_BLOCK_WRITE
#endif
#ifdef DT_INPUT_BLOCK_WRITE2
#undef DT_INPUT_BLOCK_WRITE2
#endif
#ifdef DT_INPUT_BLOCK_WRITE4
#undef DT_INPUT_BLOCK_WRITE4
#endif
#ifdef DT_INPUT_BLOCK_WRITE8
#undef DT_INPUT_BLOCK_WRITE8
#endif
#ifdef DT_INPUT_BLOCK_WRITE16
#undef DT_INPUT_BLOCK_WRITE16
#endif
#ifdef DT_OUTPUT_BLOCK_READ
#undef DT_OUTPUT_BLOCK_READ
#endif
#ifdef DT_OUTPUT_BLOCK_READ2
#undef DT_OUTPUT_BLOCK_READ2
#endif
#ifdef DT_OUTPUT_BLOCK_READ4
#undef DT_OUTPUT_BLOCK_READ4
#endif
#ifdef DT_OUTPUT_BLOCK_READ8
#undef DT_OUTPUT_BLOCK_READ8
#endif
#ifdef DT_OUTPUT_BLOCK_READ16
#undef DT_OUTPUT_BLOCK_READ16
#endif
#ifdef DT_OUTPUT_BLOCK_WRITE
#undef DT_OUTPUT_BLOCK_WRITE
#endif
#ifdef DT_OUTPUT_BLOCK_WRITE2
#undef DT_OUTPUT_BLOCK_WRITE2
#endif
#ifdef DT_OUTPUT_BLOCK_WRITE4
#undef DT_OUTPUT_BLOCK_WRITE4
#endif
#ifdef DT_OUTPUT_BLOCK_WRITE8
#undef DT_OUTPUT_BLOCK_WRITE8
#endif
#ifdef DT_OUTPUT_BLOCK_WRITE16
#undef DT_OUTPUT_BLOCK_WRITE16
#endif
#ifdef DT_BIAS_BLOCK_READ
#undef DT_BIAS_BLOCK_READ
#endif
#ifdef DT_BIAS_BLOCK_READ2
#undef DT_BIAS_BLOCK_READ2
#endif
#ifdef DT_BIAS_BLOCK_READ4
#undef DT_BIAS_BLOCK_READ4
#endif
#ifdef DT_BIAS_BLOCK_READ8
#undef DT_BIAS_BLOCK_READ8
#endif
#ifdef DT_BIAS_BLOCK_READ16
#undef DT_BIAS_BLOCK_READ16
#endif
#ifdef DT_BIAS_BLOCK_WRITE
#undef DT_BIAS_BLOCK_WRITE
#endif
#ifdef DT_BIAS_BLOCK_WRITE2
#undef DT_BIAS_BLOCK_WRITE2
#endif
#ifdef DT_BIAS_BLOCK_WRITE4
#undef DT_BIAS_BLOCK_WRITE4
#endif
#ifdef DT_BIAS_BLOCK_WRITE8
#undef DT_BIAS_BLOCK_WRITE8
#endif
#ifdef DT_BIAS_BLOCK_WRITE16
#undef DT_BIAS_BLOCK_WRITE16
#endif
#ifdef DT_FILTER_BLOCK_READ
#undef DT_FILTER_BLOCK_READ
#endif
#ifdef DT_FILTER_BLOCK_READ2
#undef DT_FILTER_BLOCK_READ2
#endif
#ifdef DT_FILTER_BLOCK_READ4
)foo", (std::string) R"foo(
#undef DT_FILTER_BLOCK_READ4
#endif
#ifdef DT_FILTER_BLOCK_READ8
#undef DT_FILTER_BLOCK_READ8
#endif
#ifdef DT_FILTER_BLOCK_READ16
#undef DT_FILTER_BLOCK_READ16
#endif
#ifdef DT_FILTER_BLOCK_WRITE
#undef DT_FILTER_BLOCK_WRITE
#endif
#ifdef DT_FILTER_BLOCK_WRITE2
#undef DT_FILTER_BLOCK_WRITE2
#endif
#ifdef DT_FILTER_BLOCK_WRITE4
#undef DT_FILTER_BLOCK_WRITE4
#endif
#ifdef DT_FILTER_BLOCK_WRITE8
#undef DT_FILTER_BLOCK_WRITE8
#endif
#ifdef DT_FILTER_BLOCK_WRITE16
#undef DT_FILTER_BLOCK_WRITE16
#endif
#ifdef MMAD_8
#undef MMAD_8
#endif
#ifdef MMAD_16
#undef MMAD_16
#endif
#ifdef MMAD_4x8
#undef MMAD_4x8
#endif
#ifdef MMAD_8x8
#undef MMAD_8x8
#endif
#ifdef MMAD_16x16
#undef MMAD_16x16
#endif
#ifdef SLM_BLOCK_WRITE_4
#undef SLM_BLOCK_WRITE_4
#endif
#ifdef SLM_BLOCK_READ_4
#undef SLM_BLOCK_READ_4
#endif
#ifdef SLM_BLOCK_READ_8
#undef SLM_BLOCK_READ_8
#endif
#ifdef BLOCK_READ_UC_1
#undef BLOCK_READ_UC_1
#endif
#ifdef BLOCK_READ_UC_2
#undef BLOCK_READ_UC_2
#endif
#ifdef BLOCK_READ_UC_4
#undef BLOCK_READ_UC_4
#endif
#ifdef BLOCK_READ_UC_8
#undef BLOCK_READ_UC_8
#endif
#ifdef BLOCK_READ_UC_16
#undef BLOCK_READ_UC_16
#endif
#ifdef BLOCK_WRITE_UC_1
#undef BLOCK_WRITE_UC_1
#endif
#ifdef BLOCK_WRITE_UC_2
#undef BLOCK_WRITE_UC_2
#endif
#ifdef BLOCK_WRITE_UC_4
#undef BLOCK_WRITE_UC_4
#endif
#ifdef BLOCK_WRITE_UC_8
#undef BLOCK_WRITE_UC_8
#endif
#ifdef BLOCK_WRITE_UC_16
#undef BLOCK_WRITE_UC_16
#endif
#undef KERNEL
#undef FUNC
#undef FUNC_CALL
#ifdef FP16_SUPPORTED
#undef FP16_SUPPORTED
#endif
#ifdef FP16_UNIT_USED
#undef FP16_UNIT_USED
#endif
#ifdef INPUT0_SIZE_X
#undef INPUT0_SIZE_X
#endif
#ifdef INPUT0_SIZE_Y
#undef INPUT0_SIZE_Y
#endif
#ifdef INPUT0_SIZE_Z
#undef INPUT0_SIZE_Z
#endif
#ifdef INPUT0_IFM_NUM
#undef INPUT0_IFM_NUM
#endif
#ifdef INPUT0_OFM_NUM
#undef INPUT0_OFM_NUM
#endif
#ifdef INPUT0_GROUPS_NUM
#undef INPUT0_GROUPS_NUM
#endif
#ifdef INPUT0_X_PITCH
#undef INPUT0_X_PITCH
#endif
#ifdef INPUT0_Y_PITCH
)foo", (std::string) R"foo(
#undef INPUT0_Y_PITCH
#endif
#ifdef INPUT0_Z_PITCH
#undef INPUT0_Z_PITCH
#endif
#ifdef INPUT0_IFM_PITCH
#undef INPUT0_IFM_PITCH
#endif
#ifdef INPUT0_OFM_PITCH
#undef INPUT0_OFM_PITCH
#endif
#ifdef INPUT0_GROUPS_PITCH
#undef INPUT0_GROUPS_PITCH
#endif
#ifdef INPUT0_OFFSET
#undef INPUT0_OFFSET
#endif
#ifdef INPUT0_VIEW_OFFSET
#undef INPUT0_VIEW_OFFSET
#endif
#ifdef INPUT0_LENGTH
#undef INPUT0_LENGTH
#endif
#ifdef INPUT0_DIMS
#undef INPUT0_DIMS
#endif
#ifdef INPUT0_SIMPLE
#undef INPUT0_SIMPLE
#endif
#ifdef INPUT0_GROUPED
#undef INPUT0_GROUPED
#endif
#ifdef INPUT0_LAYOUT_OIYX
#undef INPUT0_LAYOUT_OIYX
#endif
#ifdef INPUT0_TYPE
#undef INPUT0_TYPE
#endif
#ifdef INPUT0_VAL_MAX
#undef INPUT0_VAL_MAX
#endif
#ifdef INPUT0_VAL_MIN
#undef INPUT0_VAL_MIN
#endif
#ifdef INPUT0_VAL_ONE
#undef INPUT0_VAL_ONE
#endif
#ifdef INPUT0_VAL_ZERO
#undef INPUT0_VAL_ZERO
#endif
#ifdef TO_INPUT0_TYPE
#undef TO_INPUT0_TYPE
#endif
#ifdef TO_INPUT0_TYPE_SAT
#undef TO_INPUT0_TYPE_SAT
#endif
#ifdef AS_INPUT0_TYPE
#undef AS_INPUT0_TYPE
#endif
#ifdef INPUT0_MAX_FUNC
#undef INPUT0_MAX_FUNC
#endif
#ifdef INPUT0_MIN_FUNC
#undef INPUT0_MIN_FUNC
#endif
#ifdef INPUT0_ABS_FUNC
#undef INPUT0_ABS_FUNC
#endif
#ifdef INPUT0_TYPE_SIZE
#undef INPUT0_TYPE_SIZE
#endif
#ifdef INPUT0_IS_FP
#undef INPUT0_IS_FP
#endif
#ifdef INPUT0_SIZE
#undef INPUT0_SIZE
#endif
#ifdef INPUT0_SIZES
#undef INPUT0_SIZES
#endif
#ifdef INPUT0_PITCHES
#undef INPUT0_PITCHES
#endif
#ifdef INPUT0_PAD_BEFORE
#undef INPUT0_PAD_BEFORE
#endif
#ifdef INPUT0_PAD_AFTER
#undef INPUT0_PAD_AFTER
#endif
#ifdef INPUT0_INDEX_FUNC
)foo", (std::string) R"foo(
#undef INPUT0_INDEX_FUNC
#endif
#ifdef INIT_INPUT0_INDEX_FUNC_HERE
#undef INIT_INPUT0_INDEX_FUNC_HERE
#endif
#ifdef GET_INPUT0_OIYX_INDEX
#undef GET_INPUT0_OIYX_INDEX
#endif
#ifdef INPUT0_GET_INDEX
#undef INPUT0_GET_INDEX
#endif
#ifdef OUTPUT_SIZE_X
#undef OUTPUT_SIZE_X
#endif
#ifdef OUTPUT_SIZE_Y
#undef OUTPUT_SIZE_Y
#endif
#ifdef OUTPUT_SIZE_Z
#undef OUTPUT_SIZE_Z
#endif
#ifdef OUTPUT_IFM_NUM
#undef OUTPUT_IFM_NUM
#endif
#ifdef OUTPUT_OFM_NUM
#undef OUTPUT_OFM_NUM
#endif
#ifdef OUTPUT_GROUPS_NUM
#undef OUTPUT_GROUPS_NUM
#endif
#ifdef OUTPUT_X_PITCH
#undef OUTPUT_X_PITCH
#endif
#ifdef OUTPUT_Y_PITCH
#undef OUTPUT_Y_PITCH
#endif
#ifdef OUTPUT_Z_PITCH
#undef OUTPUT_Z_PITCH
#endif
#ifdef OUTPUT_IFM_PITCH
#undef OUTPUT_IFM_PITCH
#endif
#ifdef OUTPUT_OFM_PITCH
#undef OUTPUT_OFM_PITCH
#endif
#ifdef OUTPUT_GROUPS_PITCH
#undef OUTPUT_GROUPS_PITCH
#endif
#ifdef OUTPUT_OFFSET
#undef OUTPUT_OFFSET
#endif
#ifdef OUTPUT_VIEW_OFFSET
#undef OUTPUT_VIEW_OFFSET
#endif
#ifdef OUTPUT_LENGTH
#undef OUTPUT_LENGTH
#endif
#ifdef OUTPUT_DIMS
#undef OUTPUT_DIMS
#endif
#ifdef OUTPUT_SIMPLE
#undef OUTPUT_SIMPLE
#endif
#ifdef OUTPUT_GROUPED
#undef OUTPUT_GROUPED
#endif
#ifdef OUTPUT_LAYOUT_OS_IS_ZYX_OSV16_ISV16
#undef OUTPUT_LAYOUT_OS_IS_ZYX_OSV16_ISV16
#endif
#ifdef OUTPUT_TYPE
#undef OUTPUT_TYPE
#endif
#ifdef OUTPUT_VAL_MAX
#undef OUTPUT_VAL_MAX
#endif
#ifdef OUTPUT_VAL_MIN
#undef OUTPUT_VAL_MIN
#endif
#ifdef OUTPUT_VAL_ONE
#undef OUTPUT_VAL_ONE
#endif
#ifdef OUTPUT_VAL_ZERO
#undef OUTPUT_VAL_ZERO
#endif
#ifdef TO_OUTPUT_TYPE
#undef TO_OUTPUT_TYPE
#endif
#ifdef TO_OUTPUT_TYPE_SAT
#undef TO_OUTPUT_TYPE_SAT
#endif
#ifdef AS_OUTPUT_TYPE
#undef AS_OUTPUT_TYPE
#endif
#ifdef OUTPUT_MAX_FUNC
#undef OUTPUT_MAX_FUNC
#endif
#ifdef OUTPUT_MIN_FUNC
#undef OUTPUT_MIN_FUNC
#endif
#ifdef OUTPUT_ABS_FUNC
#undef OUTPUT_ABS_FUNC
#endif
#ifdef OUTPUT_TYPE_SIZE
#undef OUTPUT_TYPE_SIZE
#endif
#ifdef OUTPUT_IS_FP
#undef OUTPUT_IS_FP
#endif
#ifdef OUTPUT_SIZE
#undef OUTPUT_SIZE
)foo", (std::string) R"foo(
#endif
#ifdef OUTPUT_SIZES
#undef OUTPUT_SIZES
#endif
#ifdef OUTPUT_PITCHES
#undef OUTPUT_PITCHES
#endif
#ifdef OUTPUT_PAD_BEFORE
#undef OUTPUT_PAD_BEFORE
#endif
#ifdef OUTPUT_PAD_AFTER
#undef OUTPUT_PAD_AFTER
#endif
#ifdef UNIT_TYPE
#undef UNIT_TYPE
#endif
#ifdef UNIT_VAL_MAX
#undef UNIT_VAL_MAX
#endif
#ifdef UNIT_VAL_MIN
#undef UNIT_VAL_MIN
#endif
#ifdef UNIT_VAL_ONE
#undef UNIT_VAL_ONE
#endif
#ifdef UNIT_VAL_ZERO
#undef UNIT_VAL_ZERO
#endif
#ifdef TO_UNIT_TYPE
#undef TO_UNIT_TYPE
#endif
#ifdef TO_UNIT_TYPE_SAT
#undef TO_UNIT_TYPE_SAT
#endif
#ifdef AS_UNIT_TYPE
#undef AS_UNIT_TYPE
#endif
#ifdef UNIT_MAX_FUNC
#undef UNIT_MAX_FUNC
#endif
#ifdef UNIT_MIN_FUNC
#undef UNIT_MIN_FUNC
#endif
#ifdef UNIT_ABS_FUNC
#undef UNIT_ABS_FUNC
#endif
#ifdef UNIT_TYPE_SIZE
#undef UNIT_TYPE_SIZE
#endif
#ifdef UNIT_IS_FP
#undef UNIT_IS_FP
#endif
#ifdef SUB_GROUP_SIZE
#undef SUB_GROUP_SIZE
#endif
)foo"};
