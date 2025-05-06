// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline half convert_nf4_to_half_approx(uchar x) {
    half c0 = as_half((ushort)0x6C00);
    half c1 = as_half((ushort)0x9700);
    half c2 = as_half((ushort)0x6000);
    half c3 = as_half((ushort)0x6092);
    half c4 = as_half((ushort)0x8BFF);
    half c5 = as_half((ushort)0x3BFF);
    half c6 = as_half((ushort)0x3912);
    half c7 = as_half((ushort)0x2EA3);
    half c8 = as_half((ushort)0x1DA1);
    half y = as_half((ushort)x);
    half t = y * c0 + c1;
    y = t * c2;
    if (t < 0)
        y = t * c3 + c4;
    half v = c5 - y * y;
    half w = native_log2(v);
    t = c6 - w * (c7 + c8 * w);
    if (v > 0)
        y *= t;
    return y;
}

inline float convert_nf4_to_float_approx(uchar x) {
    half c0 = as_half((ushort)0x6C00);
    half c1 = as_half((ushort)0x9700);
    half c2 = as_half((ushort)0x6000);
    half c3 = as_half((ushort)0x6092);
    half c4 = as_half((ushort)0x8BFF);
    half c5 = as_half((ushort)0x3BFF);
    half c6 = as_half((ushort)0x3912);
    half c7 = as_half((ushort)0x2EA3);
    half c8 = as_half((ushort)0x1DA1);
    float y = convert_float(x);
    float t = y * c0 + c1;
    y = t * c2;
    if (t < 0)
        y = t * c3 + c4;
    float v = c5 - y * y;
    float w = native_log2(v);
    t = c6 - w * (c7 + c8 * w);
    if (v > 0)
        y *= t;
    return y;
}

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

inline half2 unpack_nf4_to_half_approx(uchar v) {
    const uchar v0 = v & 0x0F;
    const uchar v1 = (v & 0xF0) >> 4;
    return (half2)(convert_nf4_to_half_approx(v0), convert_nf4_to_half_approx(v1));
}

inline float2 unpack_nf4_to_float_approx(uchar v) {
    const uchar v0 = v & 0x0F;
    const uchar v1 = (v & 0xF0) >> 4;
    return (float2)(convert_nf4_to_float_approx(v0), convert_nf4_to_float_approx(v1));
}
