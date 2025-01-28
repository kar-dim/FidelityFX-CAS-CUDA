#include "include/CASLibWrapper.h"
#include "CASImpl.cuh"

//Implementation of the CAS DLL API
extern "C" {

    CAS_API void* CAS_initialize()
    {
        return new CASImpl();
    }

    CAS_API void CAS_supplyImage(void* casImpl, const unsigned char* inputImage, const int hasAlpha, const unsigned int rows, const unsigned int cols)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        cas->reinitializeMemory(hasAlpha, inputImage, rows, cols);
    }

    CAS_API const unsigned char* CAS_sharpenImage(void* casImpl, const int casMode, const float sharpenStrength, const float contrastAdaption)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        return cas->sharpenImage(casMode, sharpenStrength, contrastAdaption);
    }

    CAS_API void CAS_destroy(void* casImpl)
    {
        CASImpl* cas = static_cast<CASImpl*>(casImpl);
        delete cas;
    }
}