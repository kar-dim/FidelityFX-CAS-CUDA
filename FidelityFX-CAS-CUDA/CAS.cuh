#pragma once
#include <cuda_runtime.h>

__device__ inline float3 fastLerp(float3 v0, float3 v1, float t)
{
	return make_float3(fma(t, v1.x, fma(-t, v0.x, v0.x)), fma(t, v1.y, fma(-t, v0.y, v0.y)), fma(t, v1.z, fma(-t, v0.z, v0.z)));
}

__global__ void cas(cudaTextureObject_t texObj, const float sharpenStrength, const float contrastAdaption, unsigned char* casOutputR, unsigned char* casOutputG, unsigned char* casOutputB, const unsigned int height, const unsigned int width);