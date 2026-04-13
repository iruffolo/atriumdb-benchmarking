// bindings.cpp
// ============
// pybind11 glue layer
//
// Exposes all compression codecs as module-level function pairs:
//
//   <codec>_compress  (np.ndarray[int64]) -> (np.ndarray[uint8], cpp_ms: float)
//   <codec>_decompress(np.ndarray[uint8], n_samples: int)
//                                          -> (np.ndarray[int64], cpp_ms:
//                                          float)
//
// cpp_ms is measured with std::chrono::high_resolution_clock inside each
// codec's compress/decompress function.  It covers only the algorithmic work
// and excludes Python<->C++ marshalling, numpy array construction, and the
// one memcpy needed to hand the result back to Python.
//
// The Python harness records its own wall-clock time around each call with
// time.perf_counter(); the difference is the pure binding overhead.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>

#include "bzip2.cpp"
#include "codec.hpp"
#include "lzma.cpp"
#include "preprocess.hpp"
#include "zstd.cpp"
// #include "alp.cpp"

namespace py = pybind11;

static std::span<const int64_t>
to_i64(py::array_t<int64_t, py::array::c_style | py::array::forcecast> a) {
  auto info = a.request();
  return {static_cast<const int64_t *>(info.ptr),
          static_cast<size_t>(info.size)};
}

static std::span<const uint8_t>
to_u8(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a) {
  auto info = a.request();
  return {static_cast<const uint8_t *>(info.ptr),
          static_cast<size_t>(info.size)};
}

static py::array_t<uint8_t> ret_u8(std::vector<uint8_t> &&v) {
  py::array_t<uint8_t> out(static_cast<py::ssize_t>(v.size()));
  std::memcpy(out.mutable_data(), v.data(), v.size());
  return out;
}

static py::array_t<int64_t> ret_i64(std::vector<int64_t> &&v) {
  py::array_t<int64_t> out(static_cast<py::ssize_t>(v.size()));
  std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(int64_t));
  return out;
}

#define REGISTER_CODEC(m, ns, name)                                            \
  (m).def(                                                                     \
      name "_compress",                                                        \
      [](py::array_t<int64_t, py::array::c_style | py::array::forcecast> arr,  \
         preprocess::Mode mode) {                                              \
        auto res = ns::compress(to_i64(arr), mode);                            \
        return py::make_tuple(ret_u8(std::move(res.data)),                     \
                              res.cpp_compress_ms);                            \
      },                                                                       \
      py::arg("samples"), py::arg("mode"), "Compress with mode");              \
                                                                               \
  (m).def(                                                                     \
      name "_decompress",                                                      \
      [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr,  \
         size_t n_samples, preprocess::Mode mode) {                            \
        auto res = ns::decompress(to_u8(arr), n_samples, mode);                \
        return py::make_tuple(ret_i64(std::move(res.values)),                  \
                              res.cpp_decompress_ms);                          \
      },                                                                       \
      py::arg("data"), py::arg("n_samples"), py::arg("mode"),                  \
      "Decompress with mode");

PYBIND11_MODULE(compression_codecs, m) {
  m.doc() = "C++ compression codecs for the AtriumDB benchmark.\n";

  py::enum_<preprocess::Mode>(m, "Mode")
      .value("RAW", preprocess::Mode::RAW)
      .value("BYTE_SHUFFLE", preprocess::Mode::BYTE_SHUFFLE)
      .value("DPCM", preprocess::Mode::DPCM)
      .value("DOD", preprocess::Mode::DOD)
      .value("XOR", preprocess::Mode::XOR)
      .export_values();

  REGISTER_CODEC(m, zstd, "zstd")
  REGISTER_CODEC(m, bzip2, "bzip2")
  REGISTER_CODEC(m, lzma, "lzma")
  // REGISTER_CODEC(m, alp,            "alp")
}
