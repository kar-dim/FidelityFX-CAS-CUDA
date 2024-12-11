#include "include/CASLibWrapper.h"
#include "CASImpl.cuh"

extern "C" {

    CAS_API void* CAS_initialize(const unsigned int rows, const unsigned int cols)
    {
        return new CASImpl(rows, cols);
    }
    CAS_API void CAS_reinitialize(void* casImpl, const unsigned int rows, const unsigned int cols) 
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        cas->reinitializeMemory(rows, cols);
    }
    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const unsigned char* inputImage, const int casMode, const float sharpenStrength, const float contrastAdaption)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        return cas->sharpenImage(inputImage, static_cast<CASMode>(casMode), sharpenStrength, contrastAdaption);
    }
    CAS_API void CAS_destroy(void* casImpl)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        delete cas;
    }
}