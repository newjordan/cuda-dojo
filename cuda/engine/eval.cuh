// =============================================================================
// eval.cuh
//
// Tapered PeSTO eval extracted verbatim from gpu_fighter.cu's d_evaluate
// (lines 439-465). Byte-for-byte match against an exact CPU port on 6000
// sampled positions per the existing verification.
//
// Both host and device versions are exposed. They use independent storage
// for the PST tables (device: __constant__, host: regular const arrays) but
// the data is identical.
//
// `EvalFn` provides a tiny pluggable interface: pass a function pointer to a
// custom eval into search variants without recompiling them.
// =============================================================================
#ifndef ENGINE_EVAL_CUH
#define ENGINE_EVAL_CUH

#include "engine_types.h"

namespace engine {

// Pluggable evaluator type — host and device both supported.
using EvalFn       = Score (*)(const Position*);
using DeviceEvalFn = Score (*)(const Position*);

// Default PeSTO tapered evaluator.
__device__ Score d_evaluate(const Position* s);
Score            h_evaluate(const Position* s);

} // namespace engine

#endif // ENGINE_EVAL_CUH
