cmake_minimum_required(VERSION 3.22)

project(toyexample  VERSION 1.0
                  DESCRIPTION "Toy Example"
                  LANGUAGES CXX)

find_package(OpenCL REQUIRED)

add_executable(toyexample  src/main.cpp )
target_link_libraries(toyexample OpenCL::OpenCL)
target_compile_features(toyexample PRIVATE cxx_std_17)