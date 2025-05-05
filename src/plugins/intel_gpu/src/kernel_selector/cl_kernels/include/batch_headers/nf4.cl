// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline float convert_nf4_to_float(uchar v) {
    const float lookup[16] = {-1.0f,
                              -0.6961928009986877f,
                              -0.5250730514526367f,
                              -0.39491748809814453f,
                              -0.28444138169288635f,
                              -0.18477343022823334f,
                              -0.09105003625154495f,
                              0.0f,
                              0.07958029955625534f,
                              0.16093020141124725f,
                              0.24611230194568634f,
                              0.33791524171829224f,
                              0.44070982933044434f,
                              0.5626170039176941f,
                              0.7229568362236023f,
                              1.0f};

    return lookup[v];
}

inline half convert_nf4_to_half(uchar v) {
    const half lookup[16] = {-1.0h,
                              -0.6961928009986877h,
                              -0.5250730514526367h,
                              -0.39491748809814453h,
                              -0.28444138169288635h,
                              -0.18477343022823334h,
                              -0.09105003625154495h,
                              0.0h,
                              0.07958029955625534h,
                              0.16093020141124725h,
                              0.24611230194568634h,
                              0.33791524171829224h,
                              0.44070982933044434h,
                              0.5626170039176941h,
                              0.7229568362236023h,
                              1.0h};

    return lookup[v];
}

inline half2 unpack_nf4_to_half(uchar v) {
    const uchar v0 = v & 0x0F;
    const uchar v1 = (v & 0xF0) >> 4;
    return (half2)(convert_nf4_to_half(v0), convert_nf4_to_half(v1));
}

inline float2 unpack_nf4_to_float(uchar v) {
    const uchar v0 = v & 0x0F;
    const uchar v1 = (v & 0xF0) >> 4;
    return (float2)(convert_nf4_to_float(v0), convert_nf4_to_float(v1));
}
