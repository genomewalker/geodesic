#pragma once
// skani moved to genopack library — forward everything into namespace derep.
#include <genopack/skani.hpp>

namespace derep {
    using genopack::SkaniSketch;
    using genopack::SkaniResult;
    using genopack::build_sketch;
    using genopack::compute_ani;
} // namespace derep
