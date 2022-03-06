cmake_minimum_required(VERSION 3.22)

set(CMAKE_TOOLCHAIN_FILE "C:/Program Files (x86)/Android/android-sdk/ndk-bundle/build/cmake/android.toolchain.cmake")
set(ANDROID_ABI arm64-v8a)

project(toyexample  VERSION 1.0
                  DESCRIPTION "Toy Example"
                  LANGUAGES CXX)

add_executable(toyexample  src/main.cpp)