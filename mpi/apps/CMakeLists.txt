add_executable(tbsla_spmv_mpi tbsla_spmv_mpi.cpp)
target_link_libraries(tbsla_spmv_mpi PRIVATE tbsla_mpi)
# modern cmake (3.9+)
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
target_link_libraries(tbsla_spmv_mpi PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_spmv_mpi PUBLIC tbsla)
install(TARGETS tbsla_spmv_mpi DESTINATION bin)

add_executable(tbsla_perf_mpi tbsla_perf_mpi.cpp)
target_link_libraries(tbsla_perf_mpi PRIVATE tbsla_mpi)
target_link_libraries(tbsla_perf_mpi PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_perf_mpi PUBLIC tbsla)
install(TARGETS tbsla_perf_mpi DESTINATION bin)
