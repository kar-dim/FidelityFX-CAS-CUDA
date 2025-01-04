#include "CAS.cuh"
#include "CASImpl.cuh"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <type_traits>

//initialize buffers and texture data based on the provided image dimensions
void CASImpl::initializeMemory()
{
	const int channels = hasAlpha ? 4 : 3;
	//initialize CAS output buffers and pinned memory for output
	cudaMallocAsync(&casOutputBuffer, sizeof(unsigned char) * channels * rows * cols, stream);
	cudaHostAlloc((void**)&hostOutputBuffer, sizeof(unsigned char) * channels * rows * cols, cudaHostAllocDefault);
	cuda_utils::cudaStreamsSynchronize(stream);
	//initialize texture
	auto textureData = cuda_utils::createTextureData(rows, cols);
	texObj = textureData.first;
	texArray = textureData.second;
}

//full constructor
CASImpl::CASImpl(const bool hasAlpha, const unsigned int rows, const unsigned int cols) :
	hasAlpha(hasAlpha), rows(rows), cols(cols)
{
	cuda_utils::cudaStreamsCreate(stream);
	initializeMemory();
}

//copy constructor
CASImpl::CASImpl(const CASImpl& other):
	hasAlpha(other.hasAlpha), rows(other.rows), cols(other.rows)
{
	cuda_utils::cudaStreamsCreate(stream);
	initializeMemory();
}

//helper method for moving data between CAS instances
void CASImpl::moveData(CASImpl&& other) noexcept
{
	static constexpr auto moveMember = [](auto& thisData, auto& otherData) { thisData = otherData; otherData = nullptr; };
	rows = other.rows;
	cols = other.cols;
	hasAlpha = other.hasAlpha;
	//move buffer/pointer data between this and other
	moveMember(hostOutputBuffer, other.hostOutputBuffer);
	moveMember(casOutputBuffer, other.casOutputBuffer);
	moveMember(stream, other.stream);
	moveMember(texArray, other.texArray);
	//move texture object
	texObj = other.texObj;
	other.texObj = 0;
}

//move constructor
CASImpl::CASImpl(CASImpl&& other) noexcept
{
	moveData(std::move(other));
}

//move assignment
CASImpl& CASImpl::operator=(CASImpl&& other) noexcept
{
	if (this != &other)
	{
		//delete old buffers and stream and move data from other to this
		destroyBuffers();
		cuda_utils::cudaStreamsDestroy(stream);
		moveData(std::move(other));
	}
	return *this;
}

//copy assignment
CASImpl& CASImpl::operator=(const CASImpl& other)
{
	if (this != &other) 
	{
		//no need to reinitialize streams, only buffers
		reinitializeMemory(other.hasAlpha, other.rows, other.cols);
	}
	return *this;
}

//delete all buffers
void CASImpl::destroyBuffers()
{
	static constexpr auto destroy = [](auto& resource, auto& deleter) { if (resource) deleter(resource); };
	static constexpr auto destroyAsync = [](auto& resource, auto& stream, auto& deleter) { if (resource) deleter(resource, stream); };
	destroyAsync(casOutputBuffer, stream, cudaFreeAsync);
	destroy(texObj, cudaDestroyTextureObject);
	destroy(texArray, cudaFreeArray);
	destroy(hostOutputBuffer, cudaFreeHost);
	cuda_utils::cudaStreamsSynchronize(stream);
}

//destructor, destroy everything
CASImpl::~CASImpl()
{
	destroyBuffers();
	cuda_utils::cudaStreamsDestroy(stream);
}

//destory and re-initialize memory objects only
void CASImpl::reinitializeMemory(const bool hasAlpha, const unsigned int rows, const unsigned int cols)
{
	this->rows = rows;
	this->cols = cols;
	this->hasAlpha = hasAlpha;
	destroyBuffers();
	initializeMemory();
}

//setup and call main CAS kernel, return sharpened image as unsigned char buffer (pinned memory of this CAS instance)
//hostRgbPtr must be interleaved RGB(A) data
//returns the sharpened image as unsigned char buffer (planar RGB or interleaved RGBA, based on casMode param)
const unsigned char* CASImpl::sharpenImage(const unsigned char *hostRgbPtr, const int casMode, const float sharpenStrength, const float contrastAdaption)
{
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols);
	//copy input data to texture
	cuda_utils::copyDataToCudaArrayAsync(hostRgbPtr, rows, cols, texArray, stream);
	cudaStreamSynchronize(stream);
	//enqueue CAS kernel with Alpha channel output or not, or RGB planar or interleaved output based on param casMode
	if (hasAlpha && casMode == PLANAR_RGB)
		cas <true, PLANAR_RGB> << <gridSize, blockSize, 0, stream >> > (texObj, sharpenStrength, contrastAdaption, casOutputBuffer, rows, cols);
	else if (hasAlpha && casMode == INTERLEAVED_RGBA)
		cas <true, INTERLEAVED_RGBA> << <gridSize, blockSize, 0, stream >> > (texObj, sharpenStrength, contrastAdaption, casOutputBuffer, rows, cols);
	else if (!hasAlpha && casMode == PLANAR_RGB)
		cas <false, PLANAR_RGB> << <gridSize, blockSize, 0, stream >> > (texObj, sharpenStrength, contrastAdaption, casOutputBuffer, rows, cols);
	else
		cas <false, INTERLEAVED_RGBA> << <gridSize, blockSize, 0, stream >> > (texObj, sharpenStrength, contrastAdaption, casOutputBuffer, rows, cols);
	cudaStreamSynchronize(stream);
	//copy from GPU to HOST
	cudaMemcpyAsync(hostOutputBuffer, casOutputBuffer, rows * cols * sizeof(unsigned char) * (hasAlpha ? 4 : 3), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	return hostOutputBuffer;
}
