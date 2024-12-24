#pragma once
#include <cuda_runtime.h>

//Main class responsible for managing CUDA memory and calling the CAS kernel to sharpen the input image
class CASImpl
{
private:
	cudaStream_t stream;
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	unsigned char *casOutputBuffer;
	unsigned char* hostOutputBuffer;
	bool hasAlpha;
	unsigned int rows, cols;
	const dim3 blockSize { 16, 16 };

	void initializeMemory(const unsigned int rows, const unsigned int cols);
	void destroyBuffers();
public:
	CASImpl(const bool hasAlpha, const unsigned int rows, const unsigned int cols);
	CASImpl(const CASImpl& other);
	CASImpl(CASImpl&& other) noexcept;
	CASImpl& operator=(CASImpl&& other) noexcept;
	CASImpl& operator=(const CASImpl& other);
	~CASImpl();
	void reinitializeMemory(const bool hasAlpha, const unsigned int rows, const unsigned int cols);
	const unsigned char* sharpenImage(const unsigned char* inputImage, const float sharpenStrength, const float contrastAdaption);
};