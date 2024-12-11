#pragma once
#ifdef CAS_EXPORT
#define CAS_API __declspec(dllexport)
#else
#define CAS_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
    //allocate internal memory (must be the first call)
    CAS_API void* CAS_initialize(const unsigned int rows, const unsigned int cols);

    //deallocate internal memory and allocate new memory with the new specified size
    CAS_API void CAS_reinitialize(void* casImpl, const unsigned int rows, const unsigned int cols);

    //casMode = 0: return value is an RGB image pointer (R1,R2,....B1,B2,....G1,G2...) row-wise
    //casMode = 1: return value is interleaved RGB image pointer (R1,G1,B1,R2,G2,B2,...) row-wise
    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const unsigned char* inputImage, const int casMode, const float sharpenStrength, const float contrastAdaption);
    
    //free internal memory
    CAS_API void CAS_destroy(void* casImpl);

#ifdef __cplusplus
}
#endif