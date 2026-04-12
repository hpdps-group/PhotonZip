
### Generate autotuned thread table and run

```bash
cd examples/MANS
python autotune_to_csv.py
```

Expected output:

```text
compress: 985.31 MB/s, decompress: 1372.50 MB/s
thread_rows: 36
csv_path: /root/workspace/PhotonZip/build/best_threads.csv
```

### CPU roundtrip with autotuned thread table

```bash
cd examples/MANS
python cpu_roundtrip_autotune.py
```

Expected output:

```text
compress: 1011.07 MB/s, decompress: 1715.47 MB/s
csv_path: /root/workspace/PhotonZip/build/best_threads.csv, thread_rows: 36
ratio: 3.340x, restored_type: PhotonZipArray
is_equal: True
```

### NVIDIA GPU roundtrip

```bash
cd examples/MANS
python nv_roundtrip.py
```

Expected output:

```text
compress: 784.87 MB/s, decompress: 4152.21 MB/s
ratio: 2.559x, input_device: cuda:0, output_device: cuda:0
is_equal: True
```