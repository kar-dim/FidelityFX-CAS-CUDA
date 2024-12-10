#include "include/CASLibWrapper.h"
#include "CASImpl.cuh"

extern "C" {

    CAS_API void* CAS_initialize(const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption)
    {
        return new CASImpl(rows, cols, sharpenStrength, contrastAdaption);
    }
    CAS_API void CAS_reinitialize(void* casImpl, const unsigned int rows, const unsigned int cols, const float sharpenStrength, const float contrastAdaption) 
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        cas->reinitialize(rows, cols, sharpenStrength, contrastAdaption);
    }
    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const unsigned char* inputImage)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        return cas->sharpenImage(inputImage);
    }
    CAS_API void CAS_destroy(void* casImpl)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        delete cas;
    }
}