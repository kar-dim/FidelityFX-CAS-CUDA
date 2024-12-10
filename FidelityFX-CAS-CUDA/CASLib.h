#pragma once
#ifdef CAS_EXPORT
#define CAS_API __declspec(dllexport)
#else
#define CAS_API __declspec(dllimport)
#endif
