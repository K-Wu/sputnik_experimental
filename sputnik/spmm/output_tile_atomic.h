
#ifndef THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_ATOMIC_H_
#define THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_ATOMIC_H_

#include "sputnik/load_store.h"
#include "sputnik/load_store_atomic.h"
#include "sputnik/spmm/output_tile.h"
#include "sputnik/spmm/predicate_utils.h"
#include "sputnik/type_utils.h"

namespace sputnik {

template <typename Value, int kBlockItemsX, int kBlockWidth,
          bool atomicStoreFlag>
__device__ __forceinline__ void MyStore(
    OutputTile<Value, kBlockItemsX, kBlockWidth>& output_tile,
    const typename OutputTile<Value, kBlockItemsX, kBlockWidth>::Predicates&
        predicates_n) {
#pragma unroll
  for (int x_item_idx = 0; x_item_idx < output_tile.kThreadItemsX_;
       ++x_item_idx) {
    // NOTE: There are a few different ways we could have expressed
    // this loop while avoiding out-of-bounds memory accesses. See
    // the documentation for PredicateVector for more info.
    if (predicates_n.GetBit(x_item_idx)) {
      // TODO(tgale): The below branch is a hack to avoid a slight increase
      // in register usage in the float32 variants of these kernels with
      // the mixed-precision expression. Figure out a way to express this
      // without the branch and without altering the register allocation
      // for the single-precision kernels.
      if (TypeUtils<Value>::IsMixed()) {
        // Convert the accumulated results into the output representation.
        Value out;
        const int fragment_offset = x_item_idx *
                                    output_tile.kElementsPerScalar_ *
                                    output_tile.kValuesPerStore_;
        Convert(output_tile.output_fragment_ + fragment_offset, &out);
        if constexpr (atomicStoreFlag) {
          sputnik::StoreAtomic(out, output_tile.output_matrix_);
        } else {
          sputnik::Store(out, output_tile.output_matrix_);
        }
      } else {
        const Value* output_fragment =
            reinterpret_cast<const Value*>(output_tile.output_fragment_);
        if constexpr (atomicStoreFlag) {
          sputnik::StoreAtomic(output_fragment[x_item_idx],
                               output_tile.output_matrix_);
        } else {
          *(output_tile.output_matrix_) = output_fragment[x_item_idx];
        }
      }
      // Increment the pointers for the next iteration.
      output_tile.output_matrix_ += kBlockWidth;
    }
  }
}
}  // namespace sputnik
#endif  // THIRD_PARTY_SPUTNIK_SPMM_OUTPUT_TILE_ATOMIC_H_