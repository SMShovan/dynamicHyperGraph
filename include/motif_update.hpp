#ifndef MOTIF_UPDATE_HPP
#define MOTIF_UPDATE_HPP

#include "structure.hpp"
#include <vector>

// Device-side accumulator for motif delta counts (30 bins).
// Persists across subtract and add phases so both accumulate into the same buffer.
struct MotifDeltaAccumulator {
    int* d_counts = nullptr;
    void init();                         // allocates and zeros device buffer
    void readTo(std::vector<int>& out);  // copies result to host and frees
};

// Subtract motif counts for triangles touching the deleted frontier.
// Must be called BEFORE modifying data structures so old H2V/H2H are intact.
void computeMotifSubtract(const CBSTContext& h2vCtx,
                          const CBSTContext& h2hCtx,
                          const std::vector<int>& deletedIds,
                          MotifDeltaAccumulator& acc);

// Add motif counts for triangles touching the inserted frontier.
// Must be called AFTER building new H2V/H2H snapshots.
void computeMotifAdd(const CBSTContext& h2vCtx,
                     const CBSTContext& h2hCtx,
                     const std::vector<int>& insertedIds,
                     MotifDeltaAccumulator& acc);

// Convenience wrapper: calls subtract then add in one shot.
// IMPORTANT: both old and new contexts must be valid (unmodified) when called.
void computeMotifCountsDelta(const CBSTContext& oldH2vCtx,
                             const CBSTContext& oldH2hCtx,
                             const CBSTContext& newH2vCtx,
                             const CBSTContext& newH2hCtx,
                             const std::vector<int>& deletedHyperedgeIds,
                             const std::vector<int>& insertedHyperedgeIds,
                             std::vector<int>& outDeltaCounts);

#endif // MOTIF_UPDATE_HPP


