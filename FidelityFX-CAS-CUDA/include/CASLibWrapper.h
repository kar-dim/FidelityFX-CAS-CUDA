#pragma once
#ifdef CAS_EXPORT
#define CAS_API __declspec(dllexport)
#else
#define CAS_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
    CAS_API void* CAS_initialize(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption);
    CAS_API void CAS_reinitialize(void* casImpl, const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption);
    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const unsigned char* inputImage);
    CAS_API void CAS_destroy(void* casImpl);

#ifdef __cplusplus
}
#endif