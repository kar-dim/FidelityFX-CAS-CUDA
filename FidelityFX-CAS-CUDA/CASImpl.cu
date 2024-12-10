#include "CAS.cuh"
#include "CASImpl.cuh"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>

void CASImpl::initializeMemory(const unsigned int rows, const unsigned int cols)
{
	//initialize CAS output buffers and pinned memory for output
	cudaMallocAsync(&casOutputBufferR, sizeof(unsigned char) * rows * cols, streamR);
	cudaMallocAsync(&casOutputBufferG, sizeof(unsigned char) * rows * cols, streamG);
	cudaMallocAsync(&casOutputBufferB, sizeof(unsigned char) * rows * cols, streamB);
	cudaHostAlloc((void**)&pitchedBuffer, sizeof(unsigned char) * rows * cols * 3, cudaHostAllocDefault);
	cuda_utils::cudaStreamsSynchronize(streamR, streamG, streamB);
	//initialize texture
	auto textureData = cuda_utils::createTextureData(rows,cols);
	texObj = textureData.first;
	texArray = textureData.second;
}

//full constructor
CASImpl::CASImpl(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption):
	sharpenStrength(sharpenStrength), contrastAdaption(contrastAdaption), rows(rows), cols(cols)
{
	cuda_utils::cudaStreamsCreate(stream, streamR, streamG, streamB);
	initializeMemory(rows, cols);
}

//copy constructor
CASImpl::CASImpl(const CASImpl& other):
	sharpenStrength(other.sharpenStrength), contrastAdaption(other.contrastAdaption), rows(other.rows), cols(other.rows)
{
	cuda_utils::cudaStreamsCreate(stream, streamR, streamG, streamB);
	initializeMemory(rows, cols);
}

//move constructor
CASImpl::CASImpl(CASImpl&& other) noexcept :
	sharpenStrength(other.sharpenStrength), contrastAdaption(other.contrastAdaption), rows(other.rows), cols(other.rows)
{
	//move buffers and texture data and nullify other
	casOutputBufferR = other.casOutputBufferR;
	casOutputBufferG = other.casOutputBufferG;
	casOutputBufferB = other.casOutputBufferB;
	pitchedBuffer = other.pitchedBuffer;
	texObj = other.texObj;
	texArray = other.texArray;
	stream = other.stream;
	streamR = other.streamR;
	streamG = other.streamG;
	streamB = other.streamB;
	other.casOutputBufferR = nullptr;
	other.casOutputBufferG = nullptr;
	other.casOutputBufferB = nullptr;
	other.pitchedBuffer = nullptr;
	other.texObj = 0;
	other.stream = nullptr;
	other.streamR = nullptr;
	other.streamG = nullptr;
	other.streamB = nullptr;
	other.texArray = nullptr;
}

//move assignment
CASImpl& CASImpl::operator=(CASImpl&& other) noexcept
{
	if (this != &other)
	{
		sharpenStrength = other.sharpenStrength;
		contrastAdaption = other.contrastAdaption;
		rows = other.rows;
		cols = other.cols;
		//move pitched memory
		cudaFreeHost(pitchedBuffer);
		pitchedBuffer = other.pitchedBuffer;
		other.pitchedBuffer = nullptr;
		//move streams
		cuda_utils::cudaStreamsDestroy(stream, streamR, streamG, streamB);
		stream = other.stream;
		other.stream = nullptr;
		streamR = other.streamR;
		other.streamR = nullptr;
		streamG = other.streamG;
		other.streamG = nullptr;
		streamB = other.streamB;
		other.streamB = nullptr;
		//move texture object
		cudaDestroyTextureObject(texObj);
		texObj = other.texObj;
		other.texObj = 0;
		//move texture array
		cudaFreeArray(texArray);
		texArray = other.texArray;
		other.texArray = nullptr;
	}
	return *this;
}

//copy assignment
CASImpl& CASImpl::operator=(const CASImpl& other)
{
	if (this != &other)
	{
		sharpenStrength = other.sharpenStrength;
		contrastAdaption = other.contrastAdaption;
		rows = other.rows;
		cols = other.cols;
		cudaFreeHost(pitchedBuffer);
		cudaDestroyTextureObject(texObj);
		cudaFreeArray(texArray);
		initializeMemory(rows, cols);
	}
	return *this;
}

//destructor
CASImpl::~CASImpl()
{
	static constexpr auto destroy = [](auto&& resource, auto&& deleter) { if (resource) deleter(resource); };
	static constexpr auto destroyAsync = [](auto&& resource, auto &&stream, auto&& deleter) { if (resource) deleter(resource, stream); };
	destroyAsync(casOutputBufferR, streamR, cudaFreeAsync);
	destroyAsync(casOutputBufferG, streamG, cudaFreeAsync);
	destroyAsync(casOutputBufferB, streamB, cudaFreeAsync);
	cuda_utils::cudaStreamsSynchronize(streamR, streamG, streamB);
	cuda_utils::cudaStreamsDestroy(stream, streamR, streamG, streamB);
	destroy(texObj, cudaDestroyTextureObject);
	destroy(texArray, cudaFreeArray);
	destroy(pitchedBuffer, cudaFreeHost);
}

//destory and re-initialize memory objects, and setup new parameter values
void CASImpl::reinitialize(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption)
{
	this->sharpenStrength = sharpenStrength;
	this->contrastAdaption = contrastAdaption;
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(texArray);
	cudaFreeHost(pitchedBuffer);
	initializeMemory(rows, cols);
}

//setup and call main CAS kernel, return sharpened image as unsigned char buffer (pinned memory of this CAS instance)
const unsigned char* CASImpl::sharpenImage(const unsigned char *inputImage)
{
	const dim3 blockSize(16, 16);
	const dim3 gridSize = cuda_utils::gridSizeCalculate(blockSize, rows, cols);
	//copy input data to texture
	cuda_utils::copyDataToCudaArray(inputImage, rows, cols, texArray, stream);
	//enqueue CAS kernel
	cas << <gridSize, blockSize, 0, stream >> > (texObj, sharpenStrength, contrastAdaption, casOutputBufferR, casOutputBufferG, casOutputBufferB, rows, cols);
	cudaStreamSynchronize(stream);
	//copy from GPU to HOST
	cudaMemcpyAsync(pitchedBuffer, casOutputBufferR, rows * cols * sizeof(unsigned char), cudaMemcpyDefault, streamR);
	cudaMemcpyAsync(pitchedBuffer + (rows * cols), casOutputBufferG, rows * cols * sizeof(unsigned char), cudaMemcpyDefault, streamG);
	cudaMemcpyAsync(pitchedBuffer + (2 * (rows * cols)), casOutputBufferB, rows * cols * sizeof(unsigned char), cudaMemcpyDefault, streamB);
	cuda_utils::cudaStreamsSynchronize(streamR, streamG, streamB);
	return pitchedBuffer;
}
