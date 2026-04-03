#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "photonzip/core/dlpack.h"
#include "photonzip/core/codec_registry.hpp"
#include "photonzip/core/codec_types.hpp"
#include "photonzip/core/errors.hpp"

#ifdef MANS_ENABLE_NV
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

namespace photonzip {
namespace {

struct ArrayMetadata {
  DataType dtype = DataType::kUInt8;
  MemoryKind memory_kind = MemoryKind::kHost;
  std::vector<std::int64_t> shape;
};

struct PhotonZipArrayState {
  Buffer buffer;
  ArrayMetadata logical;
  ArrayMetadata original;
  std::string codec_name;
  Backend backend = Backend::kCpu;
  std::vector<std::uint32_t> codec_params;
  bool compressed = false;
};

class PhotonZipArray {
public:
  explicit PhotonZipArray(std::shared_ptr<PhotonZipArrayState> state)
      : state_(std::move(state)) {
    if (!state_) {
      throw Error("PhotonZipArray requires a valid state.");
    }
  }

  const PhotonZipArrayState& state() const {
    return *state_;
  }

  std::shared_ptr<PhotonZipArrayState> shared_state() const {
    return state_;
  }

private:
  std::shared_ptr<PhotonZipArrayState> state_;
};

class ManagedDLPackTensor {
public:
  explicit ManagedDLPackTensor(py::object tensor_like)
      : capsule_(tensor_like.attr("__dlpack__")()) {
    const char* capsule_name = PyCapsule_GetName(capsule_.ptr());
    if (!capsule_name || std::strcmp(capsule_name, "dltensor") != 0) {
      throw Error("Expected __dlpack__() to return a dltensor capsule.");
    }

    tensor_ = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule_.ptr(), "dltensor"));
    if (!tensor_) {
      throw Error("Failed to read the DLPack capsule.");
    }

    if (PyCapsule_SetName(capsule_.ptr(), "used_dltensor") != 0) {
      throw Error("Failed to mark the DLPack capsule as consumed.");
    }
  }

  ManagedDLPackTensor(const ManagedDLPackTensor&) = delete;
  ManagedDLPackTensor& operator=(const ManagedDLPackTensor&) = delete;

  ~ManagedDLPackTensor() {
    if (tensor_ && tensor_->deleter) {
      tensor_->deleter(tensor_);
    }
  }

  const DLTensor& tensor() const {
    return tensor_->dl_tensor;
  }

private:
  py::object capsule_;
  DLManagedTensor* tensor_ = nullptr;
};

std::string dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return "uint8";
    case DataType::kUInt16:
      return "uint16";
    case DataType::kUInt32:
      return "uint32";
  }
  throw Error("Unsupported PhotonZip dtype.");
}

Backend parse_backend(const std::string& value, Backend inferred_backend) {
  if (value == "auto") {
    return inferred_backend;
  }
  if (value == "cpu") {
    return Backend::kCpu;
  }
  if (value == "cuda") {
    return Backend::kCuda;
  }
  throw Error("Unsupported backend: " + value);
}

DataType parse_dtype(const DLDataType& dtype) {
  if (dtype.lanes != 1) {
    throw Error("Only scalar DLPack dtypes are supported.");
  }
  if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 8) {
    return DataType::kUInt8;
  }
  if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 16) {
    return DataType::kUInt16;
  }
  if (dtype.code == static_cast<std::uint8_t>(kDLUInt) && dtype.bits == 32) {
    return DataType::kUInt32;
  }
  throw Error("Only uint16 and uint32 DLPack tensors are supported.");
}

MemoryKind parse_memory_kind(const DLDevice& device) {
  if (device.device_type == kDLCPU) {
    return MemoryKind::kHost;
  }
  if (device.device_type == kDLCUDA) {
    return MemoryKind::kCuda;
  }
  throw Error("Only CPU and CUDA DLPack tensors are supported.");
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* what) {
  if (rhs != 0 && lhs > (std::numeric_limits<std::size_t>::max() / rhs)) {
    throw Error(std::string("Overflow while computing ") + what + ".");
  }
  return lhs * rhs;
}

std::size_t count_elements(const DLTensor& tensor) {
  if (tensor.ndim < 1 || tensor.ndim > 3) {
    throw Error("PhotonZip currently supports tensors with 1 to 3 dimensions.");
  }

  std::size_t elements = 1;
  for (int i = 0; i < tensor.ndim; ++i) {
    if (tensor.shape[i] <= 0) {
      throw Error("Tensor dimensions must be positive.");
    }
    elements = checked_product(elements, static_cast<std::size_t>(tensor.shape[i]), "tensor element count");
  }
  return elements;
}

std::size_t element_size_bytes(const DLDataType& dtype) {
  if (dtype.bits % 8 != 0) {
    throw Error("DLPack dtype bits must be byte aligned.");
  }
  return static_cast<std::size_t>(dtype.bits / 8) * static_cast<std::size_t>(dtype.lanes);
}

bool is_compact_c_contiguous(const DLTensor& tensor) {
  if (tensor.strides == nullptr) {
    return true;
  }

  std::int64_t expected_stride = 1;
  for (int i = tensor.ndim - 1; i >= 0; --i) {
    if (tensor.strides[i] != expected_stride) {
      return false;
    }
    expected_stride *= tensor.shape[i];
  }
  return true;
}

CodecOptions make_tensor_options(const std::string& codec_name,
                                 const std::string& backend,
                                 const DLTensor& tensor,
                                 std::vector<std::uint32_t> codec_params) {
  const MemoryKind memory_kind = parse_memory_kind(tensor.device);
  const Backend inferred_backend = memory_kind == MemoryKind::kCuda ? Backend::kCuda : Backend::kCpu;

  CodecOptions options;
  options.backend = parse_backend(backend, inferred_backend);
  options.dtype = parse_dtype(tensor.dtype);
  options.element_count = count_elements(tensor);
  options.shape.reserve(static_cast<std::size_t>(tensor.ndim));
  for (int i = 0; i < tensor.ndim; ++i) {
    options.shape.push_back(static_cast<std::uint32_t>(tensor.shape[i]));
  }
  options.codec_params = std::move(codec_params);
  return options;
}

InputBuffer make_input_buffer(const DLTensor& tensor) {
  if (!is_compact_c_contiguous(tensor)) {
    throw Error("PhotonZip currently requires compact C-contiguous tensors.");
  }

  InputBuffer input;
  input.memory_kind = parse_memory_kind(tensor.device);
  input.data = static_cast<const std::uint8_t*>(tensor.data) + tensor.byte_offset;
  input.size = checked_product(count_elements(tensor), element_size_bytes(tensor.dtype), "input byte size");
  return input;
}

std::vector<std::int64_t> contiguous_strides(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

CodecValue py_to_codec_value(const py::handle& obj) {
  if (obj.is_none()) {
    return CodecValue{};
  }
  if (py::isinstance<py::bool_>(obj)) {
    return CodecValue(obj.cast<bool>());
  }
  if (py::isinstance<py::int_>(obj)) {
    return CodecValue(static_cast<std::int64_t>(obj.cast<std::int64_t>()));
  }
  if (py::isinstance<py::float_>(obj)) {
    return CodecValue(obj.cast<double>());
  }
  if (py::isinstance<py::str>(obj)) {
    return CodecValue(obj.cast<std::string>());
  }
  if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    CodecList values;
    for (const auto& item : py::reinterpret_borrow<py::sequence>(obj)) {
      values.push_back(py_to_codec_value(item));
    }
    return CodecValue(std::move(values));
  }
  if (py::isinstance<py::dict>(obj)) {
    CodecDict values;
    for (const auto& item : py::reinterpret_borrow<py::dict>(obj)) {
      values.emplace(py::cast<std::string>(item.first), py_to_codec_value(item.second));
    }
    return CodecValue(std::move(values));
  }
  throw Error("Only None, bool, int, float, str, list, tuple, and dict are supported in codec requests.");
}

py::object codec_value_to_py(const CodecValue& value) {
  if (std::holds_alternative<std::monostate>(value.value)) {
    return py::none();
  }
  if (const auto* bool_value = std::get_if<bool>(&value.value)) {
    return py::bool_(*bool_value);
  }
  if (const auto* int_value = std::get_if<std::int64_t>(&value.value)) {
    return py::int_(*int_value);
  }
  if (const auto* double_value = std::get_if<double>(&value.value)) {
    return py::float_(*double_value);
  }
  if (const auto* string_value = std::get_if<std::string>(&value.value)) {
    return py::str(*string_value);
  }
  if (const auto* list_value = std::get_if<std::shared_ptr<CodecList>>(&value.value)) {
    py::list out;
    for (const auto& item : *(*list_value)) {
      out.append(codec_value_to_py(item));
    }
    return std::move(out);
  }
  if (const auto* dict_value = std::get_if<std::shared_ptr<CodecDict>>(&value.value)) {
    py::dict out;
    for (const auto& item : *(*dict_value)) {
      out[py::str(item.first)] = codec_value_to_py(item.second);
    }
    return std::move(out);
  }
  throw Error("Unsupported codec response value.");
}

DLDataType to_dlpack_dtype(DataType dtype) {
  DLDataType dl_dtype{};
  dl_dtype.code = static_cast<std::uint8_t>(kDLUInt);
  dl_dtype.lanes = 1;
  switch (dtype) {
    case DataType::kUInt8:
      dl_dtype.bits = 8;
      return dl_dtype;
    case DataType::kUInt16:
      dl_dtype.bits = 16;
      return dl_dtype;
    case DataType::kUInt32:
      dl_dtype.bits = 32;
      return dl_dtype;
  }
  throw Error("Unsupported PhotonZip dtype.");
}

DLDevice to_dlpack_device(MemoryKind memory_kind) {
  DLDevice device{};
  device.device_id = 0;
  switch (memory_kind) {
    case MemoryKind::kHost:
      device.device_type = kDLCPU;
      return device;
    case MemoryKind::kCuda:
      device.device_type = kDLCUDA;
      return device;
  }
  throw Error("Unsupported PhotonZip memory kind.");
}

std::shared_ptr<PhotonZipArrayState> make_compressed_state(const std::string& codec_name,
                                                           const CodecOptions& options,
                                                           Buffer buffer) {
  auto state = std::make_shared<PhotonZipArrayState>();
  state->buffer = std::move(buffer);
  state->logical.dtype = DataType::kUInt8;
  state->logical.memory_kind = state->buffer.memory_kind;
  state->logical.shape = {static_cast<std::int64_t>(state->buffer.size)};
  state->original.dtype = options.dtype;
  state->original.memory_kind = state->buffer.memory_kind;
  state->original.shape.reserve(options.shape.size());
  for (const auto dim : options.shape) {
    state->original.shape.push_back(static_cast<std::int64_t>(dim));
  }
  state->codec_name = codec_name;
  state->backend = options.backend;
  state->codec_params = options.codec_params;
  state->compressed = true;
  return state;
}

std::shared_ptr<PhotonZipArrayState> make_uncompressed_state(const PhotonZipArray& compressed,
                                                             Buffer buffer) {
  auto state = std::make_shared<PhotonZipArrayState>();
  state->buffer = std::move(buffer);
  state->logical = compressed.state().original;
  state->original = compressed.state().original;
  state->codec_name = compressed.state().codec_name;
  state->backend = compressed.state().backend;
  state->codec_params = compressed.state().codec_params;
  state->compressed = false;
  return state;
}

struct DLPackExportContext {
  std::shared_ptr<PhotonZipArrayState> state;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
};

void delete_dlpack_tensor(DLManagedTensor* tensor) {
  auto* context = static_cast<DLPackExportContext*>(tensor->manager_ctx);
  delete context;
  delete tensor;
}

py::capsule to_dlpack_capsule(const PhotonZipArray& array) {
#ifdef MANS_ENABLE_NV
  if (array.state().logical.memory_kind == MemoryKind::kCuda) {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      throw Error(std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(status));
    }
  }
#endif

  auto tensor = new DLManagedTensor{};
  auto* context = new DLPackExportContext{};
  context->state = array.shared_state();
  context->shape = array.state().logical.shape;
  context->strides = contiguous_strides(context->shape);

  tensor->manager_ctx = context;
  tensor->deleter = &delete_dlpack_tensor;
  tensor->dl_tensor.data = array.state().buffer.mutable_bytes();
  tensor->dl_tensor.device = to_dlpack_device(array.state().logical.memory_kind);
  tensor->dl_tensor.ndim = static_cast<int32_t>(context->shape.size());
  tensor->dl_tensor.dtype = to_dlpack_dtype(array.state().logical.dtype);
  tensor->dl_tensor.shape = context->shape.data();
  tensor->dl_tensor.strides = context->strides.empty() ? nullptr : context->strides.data();
  tensor->dl_tensor.byte_offset = 0;

  return py::capsule(tensor, "dltensor", [](PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
      auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
      if (managed && managed->deleter) {
        managed->deleter(managed);
      }
    }
  });
}

py::bytes buffer_to_py_bytes(const Buffer& buffer) {
  if (buffer.size == 0) {
    return py::bytes();
  }
  if (buffer.memory_kind == MemoryKind::kHost) {
    return py::bytes(reinterpret_cast<const char*>(buffer.bytes()), buffer.size);
  }
#ifdef MANS_ENABLE_NV
  if (buffer.memory_kind == MemoryKind::kCuda) {
    std::string host_copy(buffer.size, '\0');
    const cudaError_t status = cudaMemcpy(
        host_copy.data(), buffer.bytes(), buffer.size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      throw Error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(status));
    }
    return py::bytes(host_copy);
  }
#endif
  throw Error("Unsupported buffer memory kind.");
}

PhotonZipArray compress_tensor(const std::string& codec_name,
                               py::object input,
                               const std::string& backend,
                               std::vector<std::uint32_t> codec_params) {
  ManagedDLPackTensor managed_tensor(input);
  const auto& dl_tensor = managed_tensor.tensor();
  const auto options =
      make_tensor_options(codec_name, backend, dl_tensor, std::move(codec_params));
  const auto& codec = get_codec(codec_name);
  const auto input_buffer = make_input_buffer(dl_tensor);
  const auto output = codec.compress(input_buffer, options);
  return PhotonZipArray(make_compressed_state(codec_name, options, output));
}

PhotonZipArray decompress_tensor(const PhotonZipArray& input,
                                 const std::string& backend) {
  if (!input.state().compressed) {
    throw Error("decompress_tensor expects a compressed PhotonZipArray.");
  }

  CodecOptions options;
  options.backend = parse_backend(backend, input.state().backend);
  options.dtype = input.state().original.dtype;
  options.shape.reserve(input.state().original.shape.size());
  for (const auto dim : input.state().original.shape) {
    options.shape.push_back(static_cast<std::uint32_t>(dim));
  }
  options.codec_params = input.state().codec_params;

  InputBuffer input_buffer;
  input_buffer.data = input.state().buffer.bytes();
  input_buffer.size = input.state().buffer.size;
  input_buffer.memory_kind = input.state().buffer.memory_kind;

  const auto& codec = get_codec(input.state().codec_name);
  const auto output = codec.decompress(input_buffer, options);
  return PhotonZipArray(make_uncompressed_state(input, output));
}

py::object invoke_codec(const std::string& codec_name,
                        const std::string& op_name,
                        py::object request) {
  const auto& codec = get_codec(codec_name);
  if (!codec.invoke) {
    throw Error("Codec does not expose any extension operations: " + codec_name);
  }

  CodecInvokeRequest invoke_request;
  invoke_request.op_name = op_name;
  if (!request.is_none()) {
    const CodecValue parsed = py_to_codec_value(request);
    const auto* args = std::get_if<std::shared_ptr<CodecDict>>(&parsed.value);
    if (!args || !(*args)) {
      throw Error("codec request must be a dict.");
    }
    invoke_request.args = *(*args);
  }

  return codec_value_to_py(codec.invoke(invoke_request));
}

}  // namespace
}  // namespace photonzip

PYBIND11_MODULE(_native, module) {
  module.doc() = "PhotonZip native codec bindings";

  py::register_exception<photonzip::Error>(module, "PhotonZipError");

  py::class_<photonzip::PhotonZipArray>(module, "PhotonZipArray")
      .def_property_readonly("codec", [](const photonzip::PhotonZipArray& array) {
        return array.state().codec_name;
      })
      .def_property_readonly("compressed", [](const photonzip::PhotonZipArray& array) {
        return array.state().compressed;
      })
      .def_property_readonly("dtype", [](const photonzip::PhotonZipArray& array) {
        return photonzip::dtype_name(array.state().logical.dtype);
      })
      .def_property_readonly("shape", [](const photonzip::PhotonZipArray& array) {
        return array.state().logical.shape;
      })
      .def_property_readonly("nbytes", [](const photonzip::PhotonZipArray& array) {
        return array.state().buffer.size;
      })
      .def("to_bytes", [](const photonzip::PhotonZipArray& array) {
        return photonzip::buffer_to_py_bytes(array.state().buffer);
      })
      .def("__dlpack__",
           [](const photonzip::PhotonZipArray& array, py::args args, py::kwargs kwargs) {
             (void)args;
             (void)kwargs;
             return photonzip::to_dlpack_capsule(array);
           })
      .def("__dlpack_device__", [](const photonzip::PhotonZipArray& array) {
        const auto device = photonzip::to_dlpack_device(array.state().logical.memory_kind);
        return py::make_tuple(static_cast<int>(device.device_type), device.device_id);
      });

  module.def("list_codecs", &photonzip::list_codecs);
  module.def(
      "compress_tensor",
      &photonzip::compress_tensor,
      py::arg("codec_name"),
      py::arg("input"),
      py::arg("backend") = "auto",
      py::arg("codec_params") = std::vector<std::uint32_t>{});
  module.def(
      "decompress_tensor",
      &photonzip::decompress_tensor,
      py::arg("input"),
      py::arg("backend") = "auto");
  module.def(
      "invoke_codec",
      &photonzip::invoke_codec,
      py::arg("codec_name"),
      py::arg("op_name"),
      py::arg("request") = py::none());
}
