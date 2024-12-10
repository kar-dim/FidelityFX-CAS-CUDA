#include "CAS.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "stdio.h"
#include "helper_math.h"

//main CAS kernel (ReShade based)
__global__ void cas(cudaTextureObject_t texObj, const float sharpenStrength, const float contrastAdaption, unsigned char* casOutputR, unsigned char* casOutputG, unsigned char* casOutputB, const unsigned int height, const unsigned int width)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int outputIndex = (y * width) + x;

	if (x >= width || y >= height)
		return;

	// fetch a 3x3 neighborhood around the pixel 'e', a = a, d = d, ....i = i 
	//  a b c
	//  d(e)f
	//  g h i
	const float3 a = make_float3(tex3D<float4>(texObj, x - 1, y - 1, 0));
	const float3 b = make_float3(tex3D<float4>(texObj, x, y - 1, 0));
	const float3 c = make_float3(tex3D<float4>(texObj, x + 1, y - 1, 0));
	const float3 d = make_float3(tex3D<float4>(texObj, x - 1, y, 0));
	const float3 e = make_float3(tex3D<float4>(texObj, x, y, 0));
	const float3 f = make_float3(tex3D<float4>(texObj, x + 1, y, 0));
	const float3 g = make_float3(tex3D<float4>(texObj, x - 1, y + 1, 0));
	const float3 h = make_float3(tex3D<float4>(texObj, x, y + 1, 0));
	const float3 i = make_float3(tex3D<float4>(texObj, x + 1, y + 1, 0));

	// Soft min and max.
	//  a b c             b
	//  d e f * 0.5  +  d e f * 0.5
	//  g h i             h
	// These are 2.0x bigger (factored out the extra multiply).
	float3 mnRGB = fminf(fminf(fminf(d, e), fminf(f, b)), h);
	const float3 mnRGB2 = fminf(mnRGB, fminf(fminf(a, c), fminf(g, i)));
	mnRGB += mnRGB2;

	float3 mxRGB = fmaxf(fmaxf(fmaxf(d, e), fmaxf(f, b)), h);
	const float3 mxRGB2 = fmaxf(mxRGB, fmaxf(fmaxf(a, c), fmaxf(g, i)));
	mxRGB += mxRGB2;

	// Smooth minimum distance to signal limit divided by smooth max.
	float3 ampRGB = clamp(fminf(mnRGB, 2.0 - mxRGB) / mxRGB, 0.0f, 1.0f);

	// Shaping amount of sharpening.
	const float3 wRGB = -sqrtf(ampRGB) / (-3.0 * contrastAdaption + 8.0);

	//						  0 w 0
	//  Filter shape:		  w 1 w
	//						  0 w 0  
	const float3 filterWindow = (b + d) + (f + h);
	const float3 outColor = saturate((filterWindow * wRGB + e) / (4.0 * wRGB + 1.0));
	const float3 sharpenedValues = fastLerp(e, outColor, sharpenStrength);

	casOutputR[outputIndex] = static_cast<unsigned char>(clamp(sharpenedValues.x * 255.0f, 0.0f, 255.0f));
	casOutputG[outputIndex] = static_cast<unsigned char>(clamp(sharpenedValues.y * 255.0f, 0.0f, 255.0f));
	casOutputB[outputIndex] = static_cast<unsigned char>(clamp(sharpenedValues.z * 255.0f, 0.0f, 255.0f));
}