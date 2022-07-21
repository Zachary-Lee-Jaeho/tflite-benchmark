#!/usr/bin/bash

./linux_x86-64_benchmark_model --graph=$1 \
    --num-threads=40 --use-gpu=true \
    --report_peak_memory_footprint=true \
    --memory_footprint_check_interval_ms=5 \
    --release_dynamic_tensors=true \
    --optimize_memory_for_large_tensors=1

