#pragma once
#include <cuda_runtime.h>

class CASImpl
{
private:
	cudaStream_t stream, streamR, streamG, streamB;
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	float sharpenStrength, contrastAdaption;
	unsigned char* casOutputBufferR, *casOutputBufferG, *casOutputBufferB;
	unsigned char* pitchedBuffer;
	unsigned int rows, cols;
	void initializeMemory(const unsigned int rows, const unsigned int cols);
public:
	CASImpl(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption);
	CASImpl(const CASImpl& other);
	CASImpl(CASImpl&& other) noexcept;
	CASImpl& operator=(CASImpl&& other) noexcept;
	CASImpl& operator=(const CASImpl& other);
	~CASImpl();
	void reinitialize(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption);
	const unsigned char* sharpenImage(const unsigned char *inputImage);
};