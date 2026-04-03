#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef MANS_ENABLE_NV
#include <cuda_runtime.h>
#endif

namespace photonzip {

enum class Backend {
  kCpu,
  kCuda,
};

enum class MemoryKind {
  kHost,
  kCuda,
};

enum class DataType {
  kUInt8,
  kUInt16,
  kUInt32,
};

struct CodecOptions {
  Backend backend = Backend::kCpu;
  DataType dtype = DataType::kUInt16;
  std::size_t element_count = 0;
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> codec_params;
};

struct InputBuffer {
  const void* data = nullptr;
  std::size_t size = 0;
  MemoryKind memory_kind = MemoryKind::kHost;
};

struct Buffer {
  std::shared_ptr<void> data;
  std::size_t size = 0;
  MemoryKind memory_kind = MemoryKind::kHost;

  std::uint8_t* mutable_bytes() const {
    return static_cast<std::uint8_t*>(data.get());
  }

  const std::uint8_t* bytes() const {
    return static_cast<const std::uint8_t*>(data.get());
  }
};

inline Buffer make_host_buffer(std::size_t size) {
  Buffer buffer;
  buffer.size = size;
  buffer.memory_kind = MemoryKind::kHost;
  if (size == 0) {
    return buffer;
  }
  buffer.data = std::shared_ptr<void>(
      static_cast<void*>(new std::uint8_t[size]),
      [](void* ptr) { delete[] static_cast<std::uint8_t*>(ptr); });
  return buffer;
}

inline Buffer make_cuda_buffer(std::size_t size) {
  Buffer buffer;
  buffer.size = size;
  buffer.memory_kind = MemoryKind::kCuda;
  if (size == 0) {
    return buffer;
  }
#ifdef MANS_ENABLE_NV
  void* ptr = nullptr;
  const cudaError_t status = cudaMalloc(&ptr, size);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
  }
  buffer.data = std::shared_ptr<void>(ptr, [](void* raw_ptr) {
    if (raw_ptr) {
      cudaFree(raw_ptr);
    }
  });
  return buffer;
#else
  throw std::runtime_error("CUDA buffers require MANS_ENABLE_NV.");
#endif
}

struct CodecValue;
using CodecList = std::vector<CodecValue>;
using CodecDict = std::unordered_map<std::string, CodecValue>;

struct CodecValue {
  using Variant = std::variant<
      std::monostate,
      bool,
      std::int64_t,
      double,
      std::string,
      std::shared_ptr<CodecList>,
      std::shared_ptr<CodecDict>>;

  Variant value;

  CodecValue() = default;
  CodecValue(std::nullptr_t) : value(std::monostate{}) {}
  CodecValue(bool v) : value(v) {}
  CodecValue(std::int64_t v) : value(v) {}
  CodecValue(double v) : value(v) {}
  CodecValue(std::string v) : value(std::move(v)) {}
  CodecValue(const char* v) : value(std::string(v)) {}
  CodecValue(CodecList v) : value(std::make_shared<CodecList>(std::move(v))) {}
  CodecValue(CodecDict v) : value(std::make_shared<CodecDict>(std::move(v))) {}
};

struct CodecInvokeRequest {
  std::string op_name;
  CodecDict args;
};

using CompressFn = std::function<Buffer(
    const InputBuffer& input,
    const CodecOptions& options)>;

using DecompressFn = std::function<Buffer(
    const InputBuffer& input,
    const CodecOptions& options)>;

using MaxCompressedSizeFn = std::function<std::size_t(
    std::size_t input_nbytes,
    const CodecOptions& options)>;

using QueryDecompressedSizeFn = std::function<std::size_t(
    const void* data,
    std::size_t nbytes,
    const CodecOptions& options)>;

using CodecInvokeFn = std::function<CodecValue(
    const CodecInvokeRequest& request)>;

struct CodecVTable {
  std::string name;
  CompressFn compress;
  DecompressFn decompress;
  MaxCompressedSizeFn max_compress_size;
  QueryDecompressedSizeFn query_decompressed_size;
  CodecInvokeFn invoke;
};

}  // namespace photonzip
