# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
import os
import secrets
import shutil
import tempfile
from pathlib import Path

import torch

from ucm.store.pipeline.connector import UcmPipelineStore


def _compare_tensors(src_rows, dst_rows):
    for row_src, row_dst in zip(src_rows, dst_rows):
        for src, dst in zip(row_src, row_dst):
            if not torch.equal(src, dst):
                diff = (~torch.eq(src, dst)).sum().item()
                raise AssertionError(f"tensor mismatch: {diff} different elements")


def _make_store_config(root: Path, unique_id: str, device_id: int) -> dict[str, object]:
    tensor_size = 131072
    layer_size = 8
    chunk_size = 1
    block_size = tensor_size * layer_size * chunk_size
    return {
        "store_pipeline": "Cache|Compress|Posix",
        "storage_backends": [str(root)],
        "unique_id": unique_id,
        "timeout_ms": 10000,
        "device_id": device_id,
        "tensor_size": tensor_size,
        "shard_size": tensor_size,
        "layer_size": layer_size,
        "chunk_size": chunk_size,
        "block_size": block_size,
        "compress_ratio": 23,
        "share_buffer_enable": True,
        "buffer_number": 64,
        "waiting_queue_depth": 16,
        "running_queue_depth": 128,
        "io_direct": True,
        "stream_number": 8,
        "posix_data_trans_concurrency": 4,
        "posix_lookup_concurrency": 4,
    }


def main():
    if not torch.cuda.is_available():
        print("skip: CUDA is not available")
        return

    root = Path(tempfile.mkdtemp(prefix="ucm_cache_compress_posix_"))
    unique_id = secrets.token_hex(8)
    try:
        scheduler = UcmPipelineStore(_make_store_config(root, unique_id, -1))
        worker = UcmPipelineStore(_make_store_config(root, unique_id, 0))

        block_ids = [secrets.token_bytes(16)]
        founds = scheduler.lookup(block_ids)
        assert founds == [False], founds
        assert scheduler.lookup_on_prefix(block_ids) == -1

        tensor_size = 131072
        layer_size = 8
        src_rows = [[
            torch.arange(tensor_size // 2, dtype=torch.float32, device="cuda").to(torch.bfloat16) + i
            for i in range(layer_size)
        ]]

        for i in range(layer_size):
            task = worker.dump(block_ids, [i], [[src_rows[0][i]]])
            worker.wait(task)

        founds = scheduler.lookup(block_ids)
        assert founds == [True], founds
        assert scheduler.lookup_on_prefix(block_ids) == 0

        dst_rows = [[torch.empty_like(t) for t in src_rows[0]]]
        for i in range(layer_size):
            task = worker.load(block_ids, [i], [[dst_rows[0][i]]])
            worker.wait(task)

        _compare_tensors(src_rows, dst_rows)
        print("cache+compress+posix roundtrip passed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")
    main()
