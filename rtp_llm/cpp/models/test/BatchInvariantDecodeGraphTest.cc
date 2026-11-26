#include "rtp_llm/cpp/models/BatchInvariantDecodeGraph.h"

#include <algorithm>

using rtp_llm::batchInvariantDecodeGraphVote;
using rtp_llm::kBatchInvariantEagerVote;

constexpr int fake_rank = batchInvariantDecodeGraphVote(1, false, 1, 1, 1);
constexpr int busy_rank = batchInvariantDecodeGraphVote(8, false, 1, 5, 5);
static_assert(fake_rank == 1 && busy_rank == 8);
static_assert(std::max(fake_rank, busy_rank) == 8);

constexpr int prefill = batchInvariantDecodeGraphVote(8, true, 1, 5, 0);
constexpr int mixed   = batchInvariantDecodeGraphVote(8, true, 1, 5, 3);
static_assert(std::max(busy_rank, prefill) == kBatchInvariantEagerVote);
static_assert(std::max(fake_rank, mixed) == kBatchInvariantEagerVote);
static_assert(batchInvariantDecodeGraphVote(0, false, 1, 32, 32) == kBatchInvariantEagerVote);
static_assert(batchInvariantDecodeGraphVote(8, false, 1, 10, 5) == kBatchInvariantEagerVote);
static_assert(batchInvariantDecodeGraphVote(8, false, 2, 5, 5) == kBatchInvariantEagerVote);

int main() {}
