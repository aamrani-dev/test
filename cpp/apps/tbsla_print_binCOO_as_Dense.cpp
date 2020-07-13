#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>

#include <fstream>


int main(int argc, char** argv) {

  if(argc == 2) {
    std::ifstream is(std::string(argv[1]), std::ifstream::binary);
    tbsla::cpp::MatrixCOO mcoo;
    mcoo.read(is);
    is.close();
    mcoo.print_stats(std::cout);
    mcoo.print_infos(std::cout);
    std::cout << "--------------" << std::endl;
    std::cout << mcoo << std::endl;
    std::cout << "--------------" << std::endl;
    mcoo.print_as_dense(std::cout) << std::endl;
    std::cout << "------CSR-----" << std::endl;
    std::cout << mcoo.toCSR() << std::endl;
  }
}
