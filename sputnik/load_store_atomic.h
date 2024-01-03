
#ifndef THIRD_PARTY_SPUTNIK_LOAD_STORE_ATOMIC_H_
#define THIRD_PARTY_SPUTNIK_LOAD_STORE_ATOMIC_H_

#include <cstring>

#include "sputnik/cuda_utils.h"

namespace sputnik {

template <typename T>
__device__ __forceinline__ void StoreAtomic(const T& value, T* ptr) {
  atomicAdd(ptr, value);
}

__device__ __forceinline__ void StoreAtomic(const float4& value, float4* ptr) {
  atomicAdd(reinterpret_cast<float*>(ptr), value.x);
  atomicAdd(reinterpret_cast<float*>(ptr) + 1, value.y);
  atomicAdd(reinterpret_cast<float*>(ptr) + 2, value.z);
  atomicAdd(reinterpret_cast<float*>(ptr) + 3, value.w);
}

__device__ __forceinline__ void StoreAtomic(const float2& value, float2* ptr) {
  atomicAdd(reinterpret_cast<float*>(ptr), value.x);
  atomicAdd(reinterpret_cast<float*>(ptr) + 1, value.y);
}

__device__ __forceinline__ void StoreAtomic(const int4& value, int4* ptr) {
  atomicAdd(reinterpret_cast<int*>(ptr), value.x);
  atomicAdd(reinterpret_cast<int*>(ptr) + 1, value.y);
  atomicAdd(reinterpret_cast<int*>(ptr) + 2, value.z);
  atomicAdd(reinterpret_cast<int*>(ptr) + 3, value.w);
}

__device__ __forceinline__ void StoreAtomic(const int2& value, int2* ptr) {
  atomicAdd(reinterpret_cast<int*>(ptr), value.x);
  atomicAdd(reinterpret_cast<int*>(ptr) + 1, value.y);
}

__device__ __forceinline__ void StoreAtomic(const half8& value, half8* ptr) {
  float4 x = BitCast<float4>(value);
  StoreAtomic(x, reinterpret_cast<float4*>(ptr));
}

__device__ __forceinline__ void StoreAtomic(const half4& value, half4* ptr) {
  float2 x = BitCast<float2>(value);
  StoreAtomic(x, reinterpret_cast<float2*>(ptr));
}

__device__ __forceinline__ void StoreAtomic(const short8& value, short8* ptr) {
  int4 x = BitCast<int4>(value);
  StoreAtomic(x, reinterpret_cast<int4*>(ptr));
}

__device__ __forceinline__ void StoreAtomic(const short4& value, short4* ptr) {
  int2 x = BitCast<int2>(value);
  StoreAtomic(x, reinterpret_cast<int2*>(ptr));
}
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_LOAD_STORE_ATOMIC_H_