#ifndef THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_ATOMIC_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_ATOMIC_H_

#include "sputnik/load_store_atomic.h"
namespace sputnik {
template <int kBlockItemsX, int kBlockWidth, bool atomicStoreFlag>
__device__ __forceinline__ void MyStore(
    OutputTile<kBlockItemsX, kBlockWidth>& output_tile, int nonzeros) {
#pragma unroll
  for (int x_item_idx = 0; x_item_idx < output_tile.kThreadItemsX_;
       ++x_item_idx) {
    if (nonzeros > 0) {
      if constexpr (atomicStoreFlag) {
        sputnik::StoreAtomic(output_tile.output_fragment_[x_item_idx],
                             output_tile.output_values_);
      } else {
        sputnik::Store(output_tile.output_fragment_[x_item_idx],
                       output_tile.output_values_);
      }
    }
    nonzeros -= kBlockWidth;
    output_tile.output_values_ += kBlockWidth;
  }
}

}  // namespace sputnik
#endif  // THIRD_PARTY_SPUTNIK_SDDMM_OUTPUT_TILE_ATOMIC_H_
