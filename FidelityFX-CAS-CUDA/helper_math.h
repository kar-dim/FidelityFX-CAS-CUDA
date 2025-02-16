#pragma once
#include "cuda_runtime.h"
#include <cuda_fp16.h>

//RGB half values
//pack one half2 for vectorized 2-way operations, and one half for the last pixel
struct half3
{
    half2 x;
    half y;
};

//RGBA half value, pack two half2 for maximum vectorization 
struct half4
{
    half2 x, y;
};


////////////////////////////////////////////////////////////////////////////////
// Initialization functions
////////////////////////////////////////////////////////////////////////////////

inline __device__ half3 make_half3(const half4 x)
{
    return half3 { x.x, __low2half(x.y) };
}

inline __device__ half3 make_half3(const half2 x, const half y)
{
    return half3 { x, y };
}

inline __device__ half4 make_half4(const float4 x)
{
    return half4 { __floats2half2_rn(x.x, x.y), __floats2half2_rn(x.z, x.w) };
}

inline __device__ half3 make_half3(const float4 x)
{
    return half3 { __floats2half2_rn(x.x, x.y), __float2half(x.z) };
}


////////////////////////////////////////////////////////////////////////////////
// Math functions
////////////////////////////////////////////////////////////////////////////////

inline __device__ half3 hmin3(const half3 a, const half3 b)
{
    return half3{ __hmin2(a.x, b.x), __hmin(a.y, b.y) };
}

inline __device__ half3 hmax3(const half3 a, const half3 b)
{
    return half3{ __hmax2(a.x, b.x), __hmax(a.y, b.y) };
}

inline __device__ half powh(const half base, const half exp)
{
    return __float2half(__powf(__half2float(base), __half2float(exp)));
}

inline __device__ half3 h3rsqrtf(const half3 x)
{
    return make_half3(h2rsqrt(x.x), hrsqrt(x.y));
}

inline __device__ half3 h3rcp(const half3 x)
{
    return make_half3(h2rcp(x.x), hrcp(x.y));
}


////////////////////////////////////////////////////////////////////////////////
// Operators
////////////////////////////////////////////////////////////////////////////////

inline __device__ half3 operator-(const half3 a)
{
    return make_half3(-a.x, -a.y);
}

inline __device__ half3 operator-(const half b, const half3 a)
{
    return make_half3(__halves2half2(b, b) - a.x, b - a.y);
}

inline __device__ half3 operator+(const half3 a, const half b)
{
    return make_half3(__halves2half2(b, b) + a.x, b + a.y);
}

inline __device__ half3 operator+(const half3 a, const half3 b)
{
    return make_half3(a.x + b.x, a.y + b.y);
}

inline __device__ void operator+=(half3& a, const half3 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __device__ half operator*(const half a, const float b)
{
    return a * __float2half(b);
}

inline __device__ half3 operator*(const half3 a, const half3 b)
{
    return make_half3(a.x * b.x, a.y * b.y);
}

inline __device__ half3 operator*(const half a, const half3 b)
{
    return make_half3(b.x * __halves2half2(a, a), b.y * a);
}

inline __device__ half3 operator*(const half3 a, const half b)
{
    return make_half3(a.x * __halves2half2(b, b), a.y * b);
}


////////////////////////////////////////////////////////////////////////////////
// Various utility functions
////////////////////////////////////////////////////////////////////////////////

inline __device__ half hclamp(const half f, const half a, const half b)
{
    return __hmin(__hmax(f, a), b);
}

inline __device__ half2 hclamp2(const half2 f, const half2 a, const half2 b)
{
    return __hmin2(__hmax2(f, a), b);
}

//clamp half3 values to [0,1]
inline __device__ half3 saturateh(const half3 x)
{
    const half2 zero = half2 { CUDART_ZERO_FP16, CUDART_ZERO_FP16 };
    const half2 one = half2 { CUDART_ONE_FP16, CUDART_ONE_FP16 };
    return make_half3(hclamp2(x.x, zero, one), hclamp(x.y, CUDART_ZERO_FP16, CUDART_ONE_FP16));
}

//faster linear interpolation by using FMA operations
inline __device__ half3 lerph(const half3 v0, const half3 v1, const half t)
{
    return make_half3(__hfma2(__halves2half2(t,t), v1.x, __hfma2(__halves2half2(-t, -t), v0.x, v0.x)), __hfma(t, v1.y, __hfma(-t, v0.y, v0.y)));
}

//converts a half in the range [0,1] to an unsigned char in the range [0,255]
inline __device__ unsigned char halfToUchar(const half value)
{
    return __half2uchar_rz(value * 255.0f);
}

//Convert a linear RGB value to sRGB value
inline __device__ half sRGB(const half linearColor)
{
    const half sRGBThreshold =     __float2half(0.0031308f);  // Threshold below which the linear scale applies
    const half sRGBScale =         __float2half(12.92f);          // Scale factor for the linear region
    const half sRGBGammaScale =    __float2half(1.055f);     // Scale factor for the gamma correction region
    const half sRGBGammaExponent = __float2half(0.416667f); // Gamma correction exponent (1/2.4)
    const half sRGBOffset =        __float2half(0.055f);         // Offset for the gamma correction region
    return linearColor <= sRGBThreshold ? linearColor * sRGBScale : sRGBGammaScale * powh(linearColor, sRGBGammaExponent) - sRGBOffset;
}
