#pragma once
#ifdef CAS_EXPORT
#define CAS_API __declspec(dllexport)
#else
#define CAS_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
	//Initialize CAS instance (must be the fist function called)
    CAS_API void* CAS_initialize();

    //deallocate internal memory and allocate new memory with the new specified image size
    CAS_API void CAS_supplyImage(void* casImpl, const unsigned char* inputImage, const int hasAlpha, const unsigned int rows, const unsigned int cols);

    //sharpen the input image and return a pinned memory buffer with the sharpened RGB(A) data
    //casMode = 0: CAS kernel will write RGB planar data (RRRR....GGGG....BBBB....AAAA....)
    //casMode = 1: CAS kernel will write RGBA interleaved data (RGBA....RGBA....)
    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const int casMode, const float sharpenStrength, const float contrastAdaption);
    
    //free internal memory
    CAS_API void CAS_destroy(void* casImpl);

#ifdef __cplusplus
}
#endif