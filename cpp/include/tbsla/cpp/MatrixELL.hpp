#ifndef TBSLA_CPP_MatrixELL
#define TBSLA_CPP_MatrixELL

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixELL : public virtual Matrix {
  public:
    MatrixELL() : values(0), columns(0), max_col(0) {};
    friend std::ostream & operator<<( std::ostream &os, const MatrixELL &m);
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    int const get_nnz() {return nnz;};
    std::ostream& print(std::ostream& os) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);

  protected:
    std::vector<double> values;
    std::vector<int> columns;
    int max_col;
};

}}

#endif
