#include "CASImpl.cuh"
#include "CAS.cuh"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>

//initialize empty CAS instance
CASImpl::CASImpl() : texObj(0), texArray(nullptr), casOutputBuffer(nullptr), hostOutputBuffer(nullptr), hasAlpha(false), rows(0), cols(0), totalBytes(0)
{ }

//destructor, destroy everything
CASImpl::~CASImpl()
{
	destroyBuffers();
}

//initialize buffers and texture data based on the provided image dimensions
void CASImpl::initializeMemory()
{
	totalBytes = rows * cols * (hasAlpha ? sizeof(uchar4) : sizeof(uchar3));
	//initialize CAS output buffers and pinned memory for output
	cudaMalloc(&casOutputBuffer, totalBytes);
	cudaHostAlloc(&hostOutputBuffer, totalBytes, cudaHostAllocDefault);
	//initialize texture
	auto textureData = cuda_utils::createTextureData(rows, cols);
	texObj = textureData.first;
	texArray = textureData.second;
}

//destory and re-initialize memory objects
void CASImpl::reinitializeMemory(const bool hasAlpha, const unsigned char* hostRgbPtr, const unsigned int rows, const unsigned int cols)
{
	this->rows = rows;
	this->cols = cols;
	this->hasAlpha = hasAlpha;
	destroyBuffers();
	initializeMemory();
	cuda_utils::copyDataToCudaArray(hostRgbPtr, rows, cols, texArray);
}

//delete all buffers
void CASImpl::destroyBuffers()
{
	static constexpr auto destroy = [](auto& resource, auto& deleter) { if (resource) deleter(resource); };
	destroy(casOutputBuffer, cudaFree);
	destroy(texObj, cudaDestroyTextureObject);
	destroy(texArray, cudaFreeArray);
	destroy(hostOutputBuffer, cudaFreeHost);
}

//calls CAS kernel on the texture data, return sharpened image as unsigned char buffer (pinned memory of this CAS instance)
//overloaded method to be used when the texture data is already set (get away with one Host to Device copy if we want to sharpen the same image)
const unsigned char* CASImpl::sharpenImage(const int casMode, const float sharpenStrength, const float contrastAdaption)
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols);
	//enqueue CAS kernel with Alpha channel output or not, or RGB planar or interleaved output based on param casMode
	if (hasAlpha && casMode == PLANAR_RGB)
		cas <unsigned char, true, PLANAR_RGB> << <gridSize, blockSize >> > (texObj, sharpenStrength, contrastAdaption, reinterpret_cast<unsigned char*>(casOutputBuffer), rows, cols);
	else if (hasAlpha && casMode == INTERLEAVED_RGBA)
		cas <uchar4, true, INTERLEAVED_RGBA> << <gridSize, blockSize >> > (texObj, sharpenStrength, contrastAdaption, reinterpret_cast<uchar4*>(casOutputBuffer), rows, cols);
	else if (!hasAlpha && casMode == PLANAR_RGB)
		cas <unsigned char, false, PLANAR_RGB> << <gridSize, blockSize >> > (texObj, sharpenStrength, contrastAdaption, reinterpret_cast<unsigned char*>(casOutputBuffer), rows, cols);
	else
		cas <uchar3, false, INTERLEAVED_RGBA> << <gridSize, blockSize >> > (texObj, sharpenStrength, contrastAdaption, reinterpret_cast<uchar3*>(casOutputBuffer), rows, cols);

	//copy from GPU to HOST
	cudaMemcpy(hostOutputBuffer, casOutputBuffer, totalBytes, cudaMemcpyDeviceToHost);
	return hostOutputBuffer;
}
