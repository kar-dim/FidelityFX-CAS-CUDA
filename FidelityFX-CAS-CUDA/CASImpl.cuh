#pragma once
#include <cuda_runtime.h>

enum CASMode 
{
	CAS_RGB,
	CAS_INTERLEAVED
};

//Main class responsible for managing CUDA memory and calling the CAS kernel to sharpen the input image
class CASImpl
{
private:
	cudaStream_t stream, streamR, streamG, streamB;
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	unsigned char* casOutputBufferR, *casOutputBufferG, *casOutputBufferB, *casOutputBufferRGB;
	unsigned char* hostOutputBuffer;
	unsigned int rows, cols;
	void initializeMemory(const unsigned int rows, const unsigned int cols);
	void destroyBuffers();
public:
	CASImpl(const unsigned int rows, const unsigned int cols);
	CASImpl(const CASImpl& other);
	CASImpl(CASImpl&& other) noexcept;
	CASImpl& operator=(CASImpl&& other) noexcept;
	CASImpl& operator=(const CASImpl& other);
	~CASImpl();
	void reinitializeMemory(const unsigned int rows, const unsigned int cols);
	const unsigned char* sharpenImage(const unsigned char* inputImage, const CASMode casMode, const float sharpenStrength, const float contrastAdaption);
};