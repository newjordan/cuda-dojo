// =============================================================================
// host_compat.h
//
// Empty stand-ins for CUDA function qualifiers so the __host__ __device__
// move generator can be compiled with a host C++ compiler. nvcc must not
// include this header: under __CUDACC__ the real qualifiers come from
// cuda_runtime.h and the device build is unchanged.
// =============================================================================
#ifndef ENGINE_HOST_COMPAT_H
#define ENGINE_HOST_COMPAT_H

#if defined(__CUDACC__)
#error "host_compat.h is for host C++ only; nvcc already provides these qualifiers"
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __constant__
#define __constant__
#endif

#endif // ENGINE_HOST_COMPAT_H
