#pragma once
// codec.hpp
// =========
// Shared interface that every C++ compressor in this benchmark implements.
//
// Each codec lives in its own .cpp / namespace and exposes exactly two
// free functions:
//
//   CompressResult   compress(std::span<const int64_t> samples,
//                             preprocess::Mode mode);
//   DecompressResult decompress(std::span<const uint8_t> data,
//                               size_t n_samples
//                               preprocess::Mode mode);
//                               );
//
// Both structs carry an internal C++ timer (std::chrono, nanosecond
// resolution)

#include <chrono>
#include <cstdint>
#include <vector>

struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point t0;

  void start() { t0 = Clock::now(); }

  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
};

// Result types
//
struct CompressResult {
  std::vector<uint8_t> data; // compressed bytes
  double cpp_compress_ms;    // time inside C++ only (no Python overhead)
  double encode_ms;
  double shannon_entropy;
};

struct DecompressResult {
  std::vector<int64_t> values; // reconstructed samples
  double cpp_decompress_ms;    // time inside C++ only
  double decode_ms;
};
