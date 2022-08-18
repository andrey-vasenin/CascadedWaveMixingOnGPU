// Copied from https://github.com/lkoshale/cuda-smart-pointers
#ifndef MEMORY_H
#define MEMORY_H

#include <memory>
#include <hip/hip_runtime_api.h>

namespace hip {

#define checkErrors(ans) { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char* file, int line, bool abort = true)
{
    if (code != hipSuccess) {
        fprintf(stderr, "gpuError: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <class T>
T* hMalloc(size_t size) {
    T* ptr;
    checkErrors(hipMalloc((void**)&ptr, size));
    return ptr;
}

template <class T>
T* hMallocManaged(size_t size) {
    T* ptr;
    checkErrors(hipMallocManaged(&ptr, size));
    return ptr;
}

template <class T>
struct hDeleter {
    void operator()(T* ptr) { checkErrors(hipFree(ptr)); }
};

template <class T>
std::shared_ptr<T> shared(long long int numElements) {
    return std::shared_ptr<T>(hMalloc<T>(sizeof(T) * numElements), hDeleter<T>());
}

template <class T>
std::unique_ptr<T, hDeleter<T>> unique(long long int numElements) {
    return std::unique_ptr<T, hDeleter<T>>(hiMalloc<T>(sizeof(T) * numElements), hDeleter<T>());
}

template <class T>
std::shared_ptr<T> unified_shared(long long int numElements) {
    return std::shared_ptr<T>(hMallocManaged<T>(sizeof(T) * numElements), hDeleter<T>());
}

template <class T>
std::unique_ptr<T, hDeleter<T>> unified_unique(long long int numElements) {
    return std::unique_ptr<T, hDeleter<T>>(hMallocManaged<T>(sizeof(T) * numElements),
                                            hDeleter<T>());
}

}  // namespace hip

#endif // MEMORY_H