#pragma once

#include <dlfcn.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace finemoe {

class CudaRuntime {
    void* library_;

    template <typename T> T bind(const char* name) {
        auto address = dlsym(library_, name);
        if (!address) throw std::runtime_error(std::string("Missing CUDA runtime symbol: ") + name);
        return reinterpret_cast<T>(address);
    }

public:
    int (*copy)(void*, const void*, std::size_t, int, void*);
    int (*zero)(void*, int, std::size_t, void*);
    int (*wait)(void*, void*, unsigned int);
    int (*record)(void*, void*);
    int (*query)(void*);
    int (*create_event)(void**, unsigned int);
    int (*destroy_event)(void*);
    int (*launch)(void*, void*);
    int (*get_device)(int*);
    int (*set_device)(int);
    int (*synchronize)();
    const char* (*error)(int);

    explicit CudaRuntime(const std::string& library) {
        library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw std::runtime_error(dlerror());
        try {
            copy = bind<decltype(copy)>("cudaMemcpyAsync");
            zero = bind<decltype(zero)>("cudaMemsetAsync");
            wait = bind<decltype(wait)>("cudaStreamWaitEvent");
            record = bind<decltype(record)>("cudaEventRecord");
            query = bind<decltype(query)>("cudaEventQuery");
            create_event = bind<decltype(create_event)>("cudaEventCreateWithFlags");
            destroy_event = bind<decltype(destroy_event)>("cudaEventDestroy");
            launch = bind<decltype(launch)>("cudaGraphLaunch");
            get_device = bind<decltype(get_device)>("cudaGetDevice");
            set_device = bind<decltype(set_device)>("cudaSetDevice");
            synchronize = bind<decltype(synchronize)>("cudaDeviceSynchronize");
            error = bind<decltype(error)>("cudaGetErrorString");
        } catch (...) {
            dlclose(library_);
            throw;
        }
    }

    ~CudaRuntime() { dlclose(library_); }
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    void check(int status) const {
        if (status) throw std::runtime_error(std::string("CUDA expert cache: ") + error(status));
    }

    bool complete(void* event) const {
        const int status = query(event);
        if (status == 600) return false; // cudaErrorNotReady
        check(status);
        return true;
    }
};

class DeviceGuard {
    CudaRuntime& cuda_;
    int previous_;
    bool changed_;
public:
    DeviceGuard(CudaRuntime& cuda, int device) : cuda_(cuda) {
        cuda_.check(cuda_.get_device(&previous_));
        changed_ = previous_ != device;
        if (changed_) cuda_.check(cuda_.set_device(device));
    }
    ~DeviceGuard() { if (changed_) cuda_.set_device(previous_); }
};

} // namespace finemoe
