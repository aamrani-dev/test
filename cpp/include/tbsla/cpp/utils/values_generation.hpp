#ifndef TBSLA_CPP_VALUES_GENERATION
#define TBSLA_CPP_VALUES_GENERATION

#include <tuple>

namespace tbsla { namespace utils { namespace values_generation {
  std::tuple<int, int, double, int> cdiag_value(int i, int nv, int nr, int nc, int cdiag);
  std::tuple<int, int, double, int> cqmat_value(int i, int nr, int nc, int c, double q, unsigned int seed_mult);
}}}

#endif
