cmake_minimum_required(VERSION 3.18)
project(compression_codecs)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Python + pybind11 ─────────────────────────────────────────────────────────
execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)
pkg_check_modules(BZIP2 REQUIRED bzip2)     # dpcm_bzip2
pkg_check_modules(LZMA  REQUIRED liblzma)   # dpcm_lzma

pybind11_add_module(compression_codecs codecs/bindings.cpp)

target_include_directories(compression_codecs PRIVATE
    codecs
    ${ZSTD_INCLUDE_DIRS}
    ${BZIP2_INCLUDE_DIRS}
    ${LZMA_INCLUDE_DIRS}
)

target_link_libraries(compression_codecs PRIVATE
    ${ZSTD_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${LZMA_LIBRARIES}
)

# ── Compiler optimisations ───────────────────────────────────────────────────
# -O3 -march=native gives the C++ codecs their full speed advantage.
# Remove -march=native if building a portable binary for another machine.
target_compile_options(compression_codecs PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# ── Install target (optional) ─────────────────────────────────────────────────
# Copies the .so next to the Python scripts after build.
# Usage: cmake --install build
install(TARGETS compression_codecs
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR})
