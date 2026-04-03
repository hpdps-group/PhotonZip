#include "photonzip/codecs/mans/mans_codec.hpp"

#include <stdexcept>

#include "mans_api.hpp"
#include "photonzip/core/codec_value_utils.hpp"

#ifdef MANS_ENABLE_NV
#include <cuda_runtime.h>
#endif

namespace photonzip {
namespace {

enum class MansMode {
  kP = 0,
  kR = 1,
};

enum MansCodecParamIndex : std::size_t {
  kMode = 0,
  kAdmCompressThread = 1,
  kAdmDecompressThread = 2,
};

mans::MansAutotuneOptions parse_autotune_options(const CodecInvokeRequest& request) {
  using namespace photonzip::codec_value;
  mans::MansAutotuneOptions options;
  const CodecDict& args = request.args;

  options.data_size_mb_list = parse_double_list(args, "data_size_mb_list", options.data_size_mb_list);
  options.dims_list = parse_u32_list(args, "dims_list", options.dims_list);
  options.threads_min = static_cast<int>(parse_scalar<std::int64_t>(args, "threads_min", options.threads_min, expect_int));
  options.threads_max = static_cast<int>(parse_scalar<std::int64_t>(args, "threads_max", options.threads_max, expect_int));
  options.stride = static_cast<int>(parse_scalar<std::int64_t>(args, "stride", options.stride, expect_int));
  options.iter = static_cast<std::uint32_t>(parse_scalar<std::int64_t>(args, "iter", options.iter, expect_int));
  options.verbose = parse_scalar<bool>(args, "verbose", options.verbose, expect_bool);

  return options;
}

CodecValue encode_autotune_result(const mans::MansAutotuneOptions& options) {
  CodecList sweep_rows;
  sweep_rows.reserve(options.sweep_rows.size());
  for (const auto& row : options.sweep_rows) {
    CodecDict item;
    item.emplace("chunk_elements", static_cast<std::int64_t>(row.chunk_elements));
    item.emplace("dims", static_cast<std::int64_t>(row.dims));
    item.emplace("mode", row.mode);
    item.emplace("threads", static_cast<std::int64_t>(row.threads));
    item.emplace("throughput_mbps", row.throughput_mbps);
    sweep_rows.emplace_back(std::move(item));
  }

  CodecList best_configs;
  best_configs.reserve(options.best_configs.size());
  for (const auto& row : options.best_configs) {
    CodecDict item;
    item.emplace("chunk_elements", static_cast<std::int64_t>(row.chunk_elements));
    item.emplace("dims", static_cast<std::int64_t>(row.dims));
    item.emplace("compress_thread", static_cast<std::int64_t>(row.compress_thread));
    item.emplace("decompress_thread", static_cast<std::int64_t>(row.decompress_thread));
    best_configs.emplace_back(std::move(item));
  }

  CodecDict result;
  result.emplace("sweep_rows", std::move(sweep_rows));
  result.emplace("best_configs", std::move(best_configs));
  return result;
}

CodecValue invoke_mans_operation(const CodecInvokeRequest& request) {
  if (request.op_name == "autotune") {
    auto options = parse_autotune_options(request);
    mans::autotune(options);
    return encode_autotune_result(options);
  }

  throw std::runtime_error("Unsupported MANS codec operation: " + request.op_name);
}

std::size_t bytes_per_element(DataType dtype);

#ifdef MANS_ENABLE_NV
Buffer compress_cuda(const InputBuffer& input,
                     std::size_t element_count,
                     const mans::MansParams& params,
                     std::size_t output_capacity) {
  Buffer output = make_cuda_buffer(output_capacity);
  std::size_t output_size = output.size;

  if (input.memory_kind == MemoryKind::kCuda) {
    mans::compress_device(input.data, element_count, params, output.mutable_bytes(), output_size);
  } else if (input.memory_kind == MemoryKind::kHost) {
    const auto input_bytes = element_count * bytes_per_element(params.dtype == mans::DataType::U32 ? DataType::kUInt32 : DataType::kUInt16);
    void* d_input = nullptr;
    const cudaError_t alloc_status = cudaMalloc(&d_input, input_bytes);
    if (alloc_status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(alloc_status));
    }
    try {
      const cudaError_t copy_status = cudaMemcpy(d_input, input.data, input_bytes, cudaMemcpyHostToDevice);
      if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(copy_status));
      }
      mans::compress_device(d_input, element_count, params, output.mutable_bytes(), output_size);
    } catch (...) {
      cudaFree(d_input);
      throw;
    }
    cudaFree(d_input);
  } else {
    throw std::runtime_error("Unsupported input memory kind.");
  }

  if (output_size == 0) {
    throw std::runtime_error("MANS compression returned an empty payload. Check dims and input characteristics.");
  }
  output.size = output_size;
  return output;
}
#endif

std::uint32_t codec_param_or(const CodecOptions& options,
                             std::size_t index,
                             std::uint32_t fallback) {
  return index < options.codec_params.size() ? options.codec_params[index] : fallback;
}

mans::MansParams to_mans_params(const CodecOptions& options) {
  mans::MansParams params;
  params.backend = options.backend == Backend::kCuda ? mans::Backend::NVIDIA : mans::Backend::CPU;
  params.dtype = options.dtype == DataType::kUInt32 ? mans::DataType::U32 : mans::DataType::U16;
  const auto mans_mode = codec_param_or(options, kMode, static_cast<std::uint32_t>(MansMode::kR));
  params.mode = mans_mode == static_cast<std::uint32_t>(MansMode::kP) ? mans::Mode::P : mans::Mode::R;
  params.dims = static_cast<std::uint32_t>(options.shape.size());
  params.nx = options.shape.size() >= 1 ? options.shape[0] : 0;
  params.ny = options.shape.size() >= 2 ? options.shape[1] : 0;
  params.nz = options.shape.size() >= 3 ? options.shape[2] : 0;
  params.adm_compress_thread = codec_param_or(options, kAdmCompressThread, 32);
  params.adm_decompress_thread = codec_param_or(options, kAdmDecompressThread, 32);
  return params;
}

std::size_t bytes_per_element(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return sizeof(std::uint8_t);
    case DataType::kUInt16:
      return sizeof(std::uint16_t);
    case DataType::kUInt32:
      return sizeof(std::uint32_t);
  }
  throw std::runtime_error("Unsupported PhotonZip dtype.");
}

std::size_t infer_element_count(std::size_t nbytes, const CodecOptions& options) {
  if (options.element_count != 0) {
    return options.element_count;
  }

  const auto width = bytes_per_element(options.dtype);
  if (nbytes % width != 0) {
    throw std::runtime_error("Input size is not aligned with the requested dtype.");
  }
  return nbytes / width;
}

CodecOptions normalize_compress_options(std::size_t nbytes, const CodecOptions& options) {
  CodecOptions normalized = options;
  normalized.element_count = infer_element_count(nbytes, options);

  if (normalized.shape.empty() || normalized.shape.size() > 3) {
    throw std::runtime_error("MANS compression requires shape rank to be between 1 and 3.");
  }

  std::size_t dims_product = 1;
  for (const auto dim : normalized.shape) {
    if (dim == 0) {
      throw std::runtime_error("MANS compression requires positive shape values.");
    }
    dims_product *= static_cast<std::size_t>(dim);
  }
  if (dims_product != normalized.element_count) {
    throw std::runtime_error("MANS compression requires shape product to match the element count.");
  }
  return normalized;
}

std::size_t query_decompressed_bytes(const InputBuffer& input,
                                     const CodecOptions& options,
                                     const mans::MansParams& params) {
  if (input.memory_kind == MemoryKind::kHost) {
    return mans::get_mans_exact_decompress_bytes(input.data, input.size, params);
  }

#ifdef MANS_ENABLE_NV
  if (input.memory_kind == MemoryKind::kCuda) {
    if (input.size <= mans::kMansHeaderBytes) {
      throw std::runtime_error("MANS compressed payload is missing the header or body.");
    }
    std::vector<std::uint8_t> header_copy(mans::kMansHeaderBytes + 1);
    const cudaError_t status = cudaMemcpy(
        header_copy.data(), input.data, header_copy.size(), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy D2H header failed: ") + cudaGetErrorString(status));
    }
    return mans::get_mans_exact_decompress_bytes(header_copy.data(), input.size, params);
  }
#endif

  throw std::runtime_error("Unsupported input memory kind.");
}

}  // namespace

CodecVTable make_mans_codec() {
  CodecVTable codec;
  codec.name = "mans";

  codec.max_compress_size = [](std::size_t input_nbytes, const CodecOptions& options) -> std::size_t {
    const auto normalized = normalize_compress_options(input_nbytes, options);
    const auto params = to_mans_params(normalized);
    return mans::get_mans_max_compress_bytes(normalized.element_count, params);
  };

  codec.query_decompressed_size = [](const void* data, std::size_t nbytes, const CodecOptions& options) -> std::size_t {
    const auto params = to_mans_params(options);
    return mans::get_mans_exact_decompress_bytes(data, nbytes, params);
  };

  codec.compress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    const auto normalized = normalize_compress_options(input.size, options);
    const auto params = to_mans_params(normalized);
    if (options.backend == Backend::kCpu) {
      Buffer output = make_host_buffer(codec.max_compress_size(input.size, options));
      std::size_t output_size = output.size;
      mans::compress(input.data, normalized.element_count, params, output.mutable_bytes(), output_size);
      if (output_size == 0) {
        throw std::runtime_error("MANS compression returned an empty payload. Check dims and input characteristics.");
      }
      output.size = output_size;
      return output;
    }

#ifdef MANS_ENABLE_NV
    if (options.backend == Backend::kCuda) {
      return compress_cuda(input, normalized.element_count, params, codec.max_compress_size(input.size, options));
    }
#endif

    throw std::runtime_error("Unsupported backend.");
  };

  codec.decompress = [codec](const InputBuffer& input, const CodecOptions& options) -> Buffer {
    const auto params = to_mans_params(options);
    const auto decompressed_bytes = query_decompressed_bytes(input, options, params);

    if (options.backend == Backend::kCpu) {
      const void* input_ptr = input.data;
      Buffer compressed_host;
      if (input.memory_kind == MemoryKind::kCuda) {
        compressed_host = make_host_buffer(input.size);
        const cudaError_t copy_status = cudaMemcpy(
            compressed_host.mutable_bytes(), input.data, input.size, cudaMemcpyDeviceToHost);
        if (copy_status != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(copy_status));
        }
        input_ptr = compressed_host.bytes();
      } else if (input.memory_kind != MemoryKind::kHost) {
        throw std::runtime_error("Unsupported input memory kind.");
      }
      Buffer output = make_host_buffer(decompressed_bytes);
      std::size_t output_size = output.size;
      mans::decompress(input_ptr, input.size, params, output.mutable_bytes(), output_size);
      output.size = output_size;
      return output;
    }

#ifdef MANS_ENABLE_NV
    if (options.backend == Backend::kCuda) {
      Buffer output = make_cuda_buffer(decompressed_bytes);
      std::size_t output_size = output.size;

      if (input.memory_kind == MemoryKind::kCuda) {
        mans::decompress_device(input.data, input.size, params, output.mutable_bytes(), output_size);
      } else if (input.memory_kind == MemoryKind::kHost) {
        Buffer compressed_device = make_cuda_buffer(input.size);
        const cudaError_t copy_status = cudaMemcpy(
            compressed_device.mutable_bytes(), input.data, input.size, cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(copy_status));
        }
        mans::decompress_device(
            compressed_device.bytes(), input.size, params, output.mutable_bytes(), output_size);
      } else {
        throw std::runtime_error("Unsupported input memory kind.");
      }

      output.size = output_size;
      return output;
    }
#endif

    throw std::runtime_error("Unsupported backend.");
  };

  codec.invoke = &invoke_mans_operation;

  return codec;
}

}  // namespace photonzip
