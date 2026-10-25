# Vendored MANS

Source: https://github.com/hpdps-group/MANS
Commit: 32793bc4c029f21d8093e6f498b06041998bc067 (2026-10-06)

The MANS sources (CPU, NVIDIA, AMD and DCU backends, `mans_api`, `tools/H5Z-MANS`, `tests/`)
are vendored with the local patches listed below. Left out: `figure/`, `slurms/`, the
benchmark CSVs, `Dockerfile`, `tools/` except `tools/H5Z-MANS`, and all test data except
`testdata/u2/exafel/exafel_59200x388_64kB.u2` (used by `tests/python/test_codecs.py`).

PhotonZip builds this tree with `add_subdirectory` and `TARGET_PLATFORM=cpu` or `cpu_nv`
(the CPU and CUDA paths of the MANS codec, the CPU/CUDA `H5Z-MANS` plugin). The `amd/` and
`dcu/` backends are kept as shipped upstream but are not built; the DCU version of PhotonZip,
with its patched MANS DCU sources, is on the `dcu` branch of this repository.

Earlier PhotonZip versions used https://github.com/197895/MANS at 1a727fd as a submodule, a branch of the same
code whose changes are contained in the commit above, except for its two `tools/H5Z-MANS`
commits: 565c215 (build fix, applied below) and 1a727fd (ID checks, superseded by the
`H5Iget_type` checks below).

## Local patches

All source changes are marked `[PhotonZip]`.

`mans_api.cpp`
- Autotune generated `target_elements` samples but declared the geometry rounded up to
  whole ADM tiles (e.g. 48x48 for 2048 elements). Since this commit the CPU decoder checks
  the header geometry against the raw size, so autotune failed for every 2D/3D size off the
  tile grid. It now generates the full `nx*ny*nz` grid.

`nv/adm/adm_kernel.cuh`, `nv/adm/mapping_uint16.cu`, `nv/adm/mapping_uint32.cu`
- Port of the ADM patches of the `dcu` branch (`dcu/adm/`). Upstream packed the ADM payload
  on the host (lengths, flags, centers, codes and the lane signal scratch copied to the host,
  about a million small `memcpy` calls for 16 MiB) and decoded on the host
  (`reference::decode`). The payload is now packed on the device (`offsets_kernel`,
  `pack_meta_kernel`, `pack_signals_kernel`) and decoded by a device kernel, with an
  incremental `TileCursor` instead of two 64-bit divisions per element and full-depth 3D tiles
  staged through shared memory. The lane signal scratch is sized by the tile of the actual
  rank (2D used the 3D tile size). Codes of RAW blocks are zeroed. The U32 entry points are
  emitted by `mapping_uint16.cu`.

`nv/mans_nv.cpp`, `nv/ans/mans_nv_ans.cu`, `nv/nv_workspace.h` (new)
- Scratch memory comes from per-device, per-thread-safe workspaces (port of
  `dcu/dcu_workspace.h`) instead of `cudaMalloc`/`cudaFree` on every call.

The patched NVIDIA encoder writes streams byte-identical to the unmodified one (16 MiB EXAFEL
2D, 256 MiB and 4 MiB 3D volumes, 4 MB random 1D, 2 MB uint32 2D). Device-resident on an
RTX 5090 (compress / decompress), 16 MiB EXAFEL: 0.04 / 0.09 GB/s -> 21.0 / 92.3 GB/s.

`tools/H5Z-MANS/CMakeLists.txt`
- Include directories relative to the MANS directory instead of `CMAKE_SOURCE_DIR`, so the
  plugin builds as a subdirectory (from 197895/MANS commit 565c215).
- `OMPI_SKIP_MPICXX` / `MPICH_SKIP_MPICXX`: the plugin and its MPI test programs use the
  MPI C API and link `MPI::MPI_C`; without these, OpenMPI's `mpi.h` pulls in the C++
  bindings and the test programs fail to link.

`tools/H5Z-MANS/src/H5Z-MANS.cpp` (from the PhotonZip working copy on the RTX 5090 host)
- The chunk geometry passes the fastest-varying (last) chunk dimension as `nx`, as the
  PhotonZip codec does (see "MANS geometry" in the top-level README).
- `set_local` reads the filter parameters into a buffer large enough for both forms; a
  zero-capacity probe does not report the stored count on HDF5 1.10.
- `can_apply`/`set_local` check the ID classes with `H5Iget_type` instead of rejecting
  callback-scoped IDs for which `H5Iis_valid` reports false.
