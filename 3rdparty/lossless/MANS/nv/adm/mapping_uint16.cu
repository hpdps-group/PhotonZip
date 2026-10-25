#include "mapping_uint16.h"
#include "mapping_uint32.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nv_workspace.h"
#include "adm_kernel.cuh"
#include "adm_reference.h"

namespace mans::nv::adm {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// [PhotonZip] Port of the patched dcu backend (dcu branch). Upstream copied lengths/flags/
// centers, the per-element codes and the whole lane signal scratch to the host, packed the
// payload there (about a million small memcpy calls for 16 MiB) and copied it back, and decoded
// on the host (reference::decode): ~370 ms encode and ~190 ms decode per 16 MiB on an RTX 5090.
// Decoding now runs on the device and the payload is packed on the device with the same layout:
//   [offsets (blocks+1) x le32][centers (blocks) x le T][flags, MSB first][codes][signals]
// Only the 8-byte signal total is read back. Scratch buffers come from a reusable workspace.
constexpr unsigned int kScanThreads = 1024;
constexpr unsigned int kPackThreads = 256;

__device__ inline void store_le32(std::uint8_t* dst, std::uint32_t value) {
    dst[0] = static_cast<std::uint8_t>(value);
    dst[1] = static_cast<std::uint8_t>(value >> 8);
    dst[2] = static_cast<std::uint8_t>(value >> 16);
    dst[3] = static_cast<std::uint8_t>(value >> 24);
}

// Exclusive prefix sum of the per-block lane lengths -> offsets[0..blocks] (also written into
// the payload as le32) and the total in *total.
__global__ __launch_bounds__(kScanThreads) void offsets_kernel(const std::uint32_t* lengths, std::size_t blocks,
                                                               std::uint8_t* payload, std::uint32_t* offsets,
                                                               unsigned long long* total) {
    __shared__ unsigned long long partial[kScanThreads];
    const std::size_t tid = threadIdx.x;
    const std::size_t per_thread = (blocks + kScanThreads - 1) / kScanThreads;
    const std::size_t begin = min(tid * per_thread, blocks);
    const std::size_t end = min(begin + per_thread, blocks);
    unsigned long long local = 0;
    for (std::size_t i = begin; i < end; ++i) local += lengths[i];
    partial[tid] = local;
    __syncthreads();
    for (unsigned int offset = 1; offset < kScanThreads; offset <<= 1) {
        const unsigned long long add = tid >= offset ? partial[tid - offset] : 0ull;
        __syncthreads();
        partial[tid] += add;
        __syncthreads();
    }
    unsigned long long running = tid == 0 ? 0ull : partial[tid - 1];
    for (std::size_t i = begin; i < end; ++i) {
        offsets[i] = static_cast<std::uint32_t>(running);
        store_le32(payload + i * sizeof(std::int32_t), static_cast<std::uint32_t>(running));
        running += lengths[i];
    }
    if (tid == kScanThreads - 1) {
        const unsigned long long sum = partial[kScanThreads - 1];
        offsets[blocks] = static_cast<std::uint32_t>(sum);
        store_le32(payload + blocks * sizeof(std::int32_t), static_cast<std::uint32_t>(sum));
        *total = sum;
    }
}

template <typename T>
__global__ __launch_bounds__(kPackThreads) void pack_meta_kernel(const T* centers, const std::uint32_t* flags_words,
                                                                 std::size_t blocks, std::size_t flags_bytes,
                                                                 std::uint8_t* out_centers, std::uint8_t* out_flags) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < blocks) kernel::store_le<T>(out_centers + i * sizeof(T), centers[i]);
    if (i < flags_bytes) {
        std::uint8_t byte = 0;
        for (unsigned int k = 0; k < 8; ++k) {
            const std::size_t block = i * 8 + k;
            if (block < blocks && (flags_words[block >> 5] & (1u << (31 - (block & 31))))) {
                byte |= static_cast<std::uint8_t>(1u << (7 - k));
            }
        }
        out_flags[i] = byte;
    }
}

// Block b's 32 lane signals (each lengths[b] bytes, padded to lane_stride in scratch) are stored
// back to back at signals + offsets[b] * 32.
__global__ __launch_bounds__(kPackThreads) void pack_signals_kernel(const std::uint8_t* lane_signals,
                                                                    std::size_t lane_stride,
                                                                    const std::uint32_t* lengths,
                                                                    const std::uint32_t* offsets,
                                                                    std::uint8_t* out_signals) {
    const std::size_t block = blockIdx.x;
    const std::uint32_t length = lengths[block];
    const std::uint32_t bytes = length * reference::kWarpSize;
    std::uint8_t* dst = out_signals + static_cast<std::size_t>(offsets[block]) * reference::kWarpSize;
    const std::uint8_t* src = lane_signals + block * reference::kWarpSize * lane_stride;
    for (std::uint32_t idx = threadIdx.x; idx < bytes; idx += blockDim.x) {
        const std::uint32_t lane = idx / length;
        dst[idx] = src[lane * lane_stride + (idx - lane * length)];
    }
}

// Shared-memory staging pays off for full-depth 3D tiles (16x16x16, 128 elements per lane spread
// over 8 rows); for thin volumes (nz < 16) the lost occupancy costs more than it saves.
bool stage_tiles(const mans::MansParams& params) {
    return params.dims == 3 && params.nz >= reference::kTileZ;
}

template <typename T>
void compress_kernel(const T* d_input, std::size_t elements,
                     const mans::MansParams& params, std::uint8_t* d_output,
                     std::size_t& output_size, cudaStream_t stream) {
    if (!d_input || !d_output || elements == 0) throw std::runtime_error("CUDA ADM input/output is null");
    const std::size_t blocks = reference::block_count(elements, params);
    const std::size_t flags_bytes = reference::flags_size(blocks);
    const std::size_t flags_words = (flags_bytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
    const std::size_t max_block = params.dims == 1 ? reference::kBlock1D :
        params.dims == 2 ? reference::kTileX * reference::kTileY : reference::kTileX * reference::kTileY * reference::kTileZ;
    const std::size_t lane_stride = reference::ceil_div(max_block, reference::kWarpSize) *
        reference::max_adm_signal_bytes_per_element();

    WorkspaceLease ws(kAdmPool);
    auto* flags = ws->get<std::uint32_t>(0, flags_words);
    auto* centers = ws->get<T>(1, blocks);
    auto* lengths = ws->get<std::uint32_t>(2, blocks);
    auto* codes = ws->get<std::uint8_t>(3, elements);
    auto* signals = ws->get<std::uint8_t>(4, blocks * reference::kWarpSize * lane_stride);
    auto* offsets = ws->get<std::uint32_t>(5, blocks + 1);
    auto* total = ws->get<unsigned long long>(6, 1);
    check_cuda(cudaMemsetAsync(flags, 0, flags_words * sizeof(std::uint32_t), stream), "CUDA ADM flags memset");
    // Codes of RAW blocks are never written by the encoder; zero them so the payload is
    // deterministic (upstream left uninitialised allocation contents there).
    check_cuda(cudaMemsetAsync(codes, 0, elements, stream), "CUDA ADM codes memset");
    if (stage_tiles(params)) {
        kernel::encode<T, true><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
            d_input, elements, params, codes, flags, centers, lengths, signals, lane_stride);
    } else {
        kernel::encode<T, false><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
            d_input, elements, params, codes, flags, centers, lengths, signals, lane_stride);
    }
    offsets_kernel<<<1, kScanThreads, 0, stream>>>(lengths, blocks, d_output, offsets, total);
    check_cuda(cudaGetLastError(), "CUDA ADM encode launch");

    unsigned long long host_total = 0;
    check_cuda(cudaMemcpyAsync(&host_total, total, sizeof(host_total), cudaMemcpyDeviceToHost, stream), "CUDA ADM total D2H");
    check_cuda(cudaStreamSynchronize(stream), "CUDA ADM encode synchronize");
    if (host_total > static_cast<unsigned long long>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("CUDA ADM signal offsets exceed int32");
    }

    const std::size_t offsets_bytes = (blocks + 1) * sizeof(std::int32_t);
    const std::size_t centers_bytes = blocks * sizeof(T);
    const std::size_t signal_bytes = static_cast<std::size_t>(host_total) * reference::kWarpSize;
    output_size = offsets_bytes + centers_bytes + flags_bytes + elements + signal_bytes;
    std::uint8_t* out_centers = d_output + offsets_bytes;
    std::uint8_t* out_flags = out_centers + centers_bytes;
    std::uint8_t* out_codes = out_flags + flags_bytes;
    const std::size_t meta_items = std::max(blocks, flags_bytes);
    pack_meta_kernel<T><<<static_cast<unsigned int>((meta_items + kPackThreads - 1) / kPackThreads), kPackThreads, 0, stream>>>(
        centers, flags, blocks, flags_bytes, out_centers, out_flags);
    check_cuda(cudaMemcpyAsync(out_codes, codes, elements, cudaMemcpyDeviceToDevice, stream), "CUDA ADM codes D2D");
    if (signal_bytes != 0) {
        pack_signals_kernel<<<static_cast<unsigned int>(blocks), kPackThreads, 0, stream>>>(
            signals, lane_stride, lengths, offsets, out_codes + elements);
    }
    check_cuda(cudaGetLastError(), "CUDA ADM pack launch");
    check_cuda(cudaStreamSynchronize(stream), "CUDA ADM pack synchronize");
}

template <typename T>
void decompress_kernel(const std::uint8_t* d_input, std::size_t input_size,
                       T* d_output, std::size_t elements,
                       const mans::MansParams& params, cudaStream_t stream) {
    if (!d_input || !d_output || elements == 0) throw std::runtime_error("CUDA ADM input/output is null");
    const std::size_t blocks = reference::block_count(elements, params);
    const std::size_t fixed = (blocks + 1) * sizeof(std::int32_t) + blocks * sizeof(T) +
                              reference::flags_size(blocks) + elements;
    if (input_size < fixed) throw std::runtime_error("CUDA ADM payload is truncated");

    WorkspaceLease ws(kAdmPool);
    auto* error = ws->get<std::uint32_t>(7, 1);
    check_cuda(cudaMemsetAsync(error, 0, sizeof(std::uint32_t), stream), "CUDA ADM error memset");
    constexpr unsigned int kValidationThreads = 256;
    const unsigned int validation_grid = static_cast<unsigned int>(
        (blocks + kValidationThreads - 1) / kValidationThreads);
    if (validation_grid == 0) throw std::runtime_error("CUDA ADM has no blocks");
    kernel::validate_payload<<<validation_grid, kValidationThreads, 0, stream>>>(
        d_input, input_size, blocks, fixed,
        error);
    check_cuda(cudaGetLastError(), "CUDA ADM payload validation launch");
    check_cuda(cudaStreamSynchronize(stream), "CUDA ADM payload validation synchronize");
    std::uint32_t host_error = 0;
    check_cuda(cudaMemcpy(&host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost), "CUDA ADM validation result");
    if (host_error != 0) throw std::runtime_error("CUDA ADM payload offsets are invalid");

    if (stage_tiles(params)) {
        kernel::decode<T, true><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
            d_input, input_size, elements, params, d_output, error);
    } else {
        kernel::decode<T, false><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
            d_input, input_size, elements, params, d_output, error);
    }
    check_cuda(cudaGetLastError(), "CUDA ADM decode launch");
    check_cuda(cudaStreamSynchronize(stream), "CUDA ADM decode synchronize");
    check_cuda(cudaMemcpy(&host_error, error, sizeof(host_error), cudaMemcpyDeviceToHost), "CUDA ADM decode result");
    if (host_error != 0) throw std::runtime_error("CUDA ADM payload decode failed");
}

} // namespace

void compress_u16_device(const std::uint16_t* d_input, std::size_t n, const mans::MansParams& p,
                         std::uint8_t* d_output, std::size_t& size, cudaStream_t stream) {
    compress_kernel(d_input, n, p, d_output, size, stream);
}
void decompress_u16_device(const std::uint8_t* d_input, std::size_t size, std::uint16_t* d_output,
                           std::size_t n, const mans::MansParams& p, cudaStream_t stream) {
    decompress_kernel(d_input, size, d_output, n, p, stream);
}
std::size_t get_max_u16_payload_bytes(std::size_t n, const mans::MansParams& p) {
    return reference::max_payload_bytes<std::uint16_t>(n, p);
}

void compress_u32_device(const std::uint32_t* d_input, std::size_t n, const mans::MansParams& p,
                         std::uint8_t* d_output, std::size_t& size, cudaStream_t stream) {
    compress_kernel(d_input, n, p, d_output, size, stream);
}
void decompress_u32_device(const std::uint8_t* d_input, std::size_t size, std::uint32_t* d_output,
                           std::size_t n, const mans::MansParams& p, cudaStream_t stream) {
    decompress_kernel(d_input, size, d_output, n, p, stream);
}
std::size_t get_max_u32_payload_bytes(std::size_t n, const mans::MansParams& p) {
    return reference::max_payload_bytes<std::uint32_t>(n, p);
}

} // namespace mans::nv::adm
