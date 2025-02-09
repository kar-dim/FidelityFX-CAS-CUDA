#pragma once
#include <cuda_runtime.h>

enum CASMode 
{
	PLANAR_RGB,
	INTERLEAVED_RGBA
};

//Main class responsible for managing CUDA memory and calling the CAS kernel to sharpen the input image
class CASImpl
{
private:
	cudaTextureObject_t texObj;
	cudaArray* texArray;
	void* casOutputBuffer;
	unsigned char* hostOutputBuffer;
	bool hasAlpha;
	unsigned int rows, cols;
	unsigned long long totalBytes;
	const dim3 blockSize { 16, 16 };

	void initializeMemory();
	void destroyBuffers();

public:
	CASImpl();
	~CASImpl();

	//delete move/copy ctors/operators, not useful for a DLL class
	CASImpl(const CASImpl& other) = delete;
	CASImpl(CASImpl&& other) noexcept = delete;
	CASImpl& operator=(CASImpl&& other) noexcept = delete;
	CASImpl& operator=(const CASImpl& other) = delete;

	void reinitializeMemory(const bool hasAlpha, const unsigned char* hostRgbPtr, const unsigned int rows, const unsigned int cols);
	const unsigned char* sharpenImage(const int casMode, const float sharpenStrength, const float contrastAdaption);
};