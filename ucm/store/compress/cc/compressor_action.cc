#include <algorithm>
#include "logger/logger.h"
#include "compressor_action.h"




namespace UC::Compressor {

std::unique_ptr<MemoryPool> CompressorAction::MemoryPoolCache::Acquire(size_t blockSize,
                                                                       size_t poolSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t alignedBlockSize = (blockSize + 4095) & ~static_cast<size_t>(4095);
    for (auto it = pools_.begin(); it != pools_.end(); ++it) {
        auto& pool = *it;
        if (pool->BlockSize() == alignedBlockSize && pool->Capacity() >= poolSize &&
            pool->Available() == pool->Capacity()) {
            auto reused = std::move(pool);
            pools_.erase(it);
            return reused;
        }
    }
    return std::make_unique<MemoryPool>(blockSize, poolSize);
}

void CompressorAction::MemoryPoolCache::Release(std::unique_ptr<MemoryPool> pool)
{
    if (!pool) { return; }
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.push_back(std::move(pool));
}

CompressorAction::~CompressorAction()
{
    // 后续改多线程需要在这销毁线程
}

Status CompressorAction::Setup(const Config& config, TaskIdSet* failureSet) 
{
    backend_ = config.storeBackend;
    failureSet_ = failureSet;
    shardSize_ = config.shardSize;
    switch (config.compressRatio) {
        case 32: ratio = R1; break;
        case 24: ratio = R133; break;
        case 23: ratio = R139; break;
        case 22: ratio = R145; break;
        case 21: ratio = R152; break;
        default: return Status::InvalidParam("invalid compressRatio({})", config.compressRatio);
    }

    switch (config.dataType) {
        case 0: dataType = DT_BF16; break;
        default: return Status::InvalidParam("invalid compress dataType({})", config.dataType);
    }
    
    // init thread pool
    dump_pool_.SetNWorker(config.streamNumber/2)
              .SetWorkerFn([this](auto& ct, auto&) { Compress_Dump(ct); })
              .Run();
    load_pool_.SetNWorker(config.streamNumber/2)
              .SetWorkerFn([this](auto& ct, auto&) { Compress_Load(ct); })
              .Run();
    decompress_pool_.SetNWorker(std::max<size_t>(1, config.streamNumber))
                    .SetWorkerFn([this](auto& dt, auto&) { Decompress_OneShard(dt); })
                    .Run();

    return Status::OK();
}

void CompressorAction::Push(TaskPtr task, WaiterPtr waiter)
{
    UC_DEBUG("task {}, push size is {}", task->id, task->desc.size());
    waiter->Set(1);
    if (task->type == TransTask::Type::DUMP) {
        dump_pool_.Push(CompressTask {
            task,
            waiter
        });
    } else {
        load_pool_.Push(CompressTask {
            task,
            waiter
        });
    }
}

void CompressorAction::Compress_Load(CompressTask& ct)
{
#ifdef USE_C_COMPRESS
    UC_DEBUG("COMPRESS LOAD START, task {}", ct.task->id);

    auto markFailure = [&](const Status& s) {
        UC_ERROR("COMPRESS LOAD failed({}) on task {}.", s, ct.task->id);
        if (failureSet_) { failureSet_->Insert(ct.task->id); }
    };

    const auto& desc = ct.task->desc;
    if (desc.empty()) {
        UC_DEBUG("COMPRESS LOAD desc is empty...");
        ct.waiter->Done();
        return;
    }

    size_t srcSize = (shardSize_ * (size_t)ratio / 32) / 4096 * 4096;
    constexpr size_t kTargetBatchBytes = 4 * 1024 * 1024;
    constexpr size_t kMaxBatchShards = 16;
    const size_t batchShardCount = std::min<size_t>(
        desc.size(),
        std::clamp(kTargetBatchBytes / std::max<size_t>(shardSize_, 1UL), size_t{1},
                   kMaxBatchShards));

    struct LoadBatch {
        Detail::TaskDesc backendDesc;
        std::vector<const Detail::Shard*> dstShards;
        std::vector<void*> compressedBuffers;
        MemoryPool* pool{nullptr};
        size_t begin{0};
        size_t end{0};
    };

    auto prepareBatch = [&](LoadBatch& batch, MemoryPool& pool, size_t begin) -> size_t {
        batch.backendDesc.clear();
        batch.backendDesc.brief = desc.brief;
        batch.dstShards.clear();
        batch.compressedBuffers.clear();
        batch.pool = &pool;
        batch.begin = begin;
        batch.end = std::min(begin + batchShardCount, desc.size());

        for (size_t i = batch.begin; i < batch.end; ++i) {
            const auto& shard = desc[i];
            uint8_t* compBuf = static_cast<uint8_t*>(pool.allocate());
            if (compBuf == nullptr) { return batch.begin; }
            batch.compressedBuffers.push_back(static_cast<void*>(compBuf));
            batch.dstShards.push_back(&shard);
            batch.backendDesc.push_back(Detail::Shard {
                shard.owner,
                shard.index,
                {static_cast<void*>(compBuf)}
            });
        }
        return batch.end;
    };

    auto releaseBatch = [](LoadBatch& batch) {
        if (batch.pool && !batch.compressedBuffers.empty()) {
            batch.pool->deallocate(batch.compressedBuffers);
        }
        batch.backendDesc.clear();
        batch.dstShards.clear();
        batch.compressedBuffers.clear();
        batch.pool = nullptr;
        batch.begin = 0;
        batch.end = 0;
    };

    auto waitBackendLoad = [&](Expected<Detail::TaskHandle>& result,
                               const LoadBatch& batch) -> Status {
        if (!result) {
            UC_ERROR("Failed({}) to submit backend load for task {} batch [{}:{})", result.Error(),
                     ct.task->id, batch.begin, batch.end);
            return result.Error();
        }
        UC_DEBUG("load result {} batch [{}:{})", result.Value(), batch.begin, batch.end);
        if (result.Value() == 0) { return Status::Error("invalid backend load task handle"); }
        auto s = backend_->Wait(result.Value());
        if (s.Failure()) {
            UC_ERROR("Failed({}) to wait backend load({}) for task {} batch [{}:{})", s,
                     result.Value(), ct.task->id, batch.begin, batch.end);
        }
        return s;
    };

    auto decompressBatch = [&](LoadBatch& batch, size_t& totalBytes) -> Status {
        if (batch.dstShards.size() != batch.compressedBuffers.size()) {
            return Status::Error("load batch buffer metadata mismatch");
        }
        auto group = std::make_shared<DecompressGroup>();
        group->waiter.Set(batch.dstShards.size());
        for (size_t i = 0; i < batch.dstShards.size(); ++i) {
            const UC::Detail::Shard& shard = *batch.dstShards[i];
            if (shard.addrs.empty() || shard.addrs[0] == nullptr) {
                releaseBatch(batch);
                return Status::InvalidParam("invalid load dst addr for shard({})", shard.index);
            }
            decompress_pool_.Push(DecompressShardTask {
                ct.task->id,
                shard.index,
                batch.compressedBuffers[i],
                shard.addrs[0],
                srcSize,
                shardSize_,
                ratio,
                group
            });
        }
        group->waiter.Wait();
        releaseBatch(batch);
        if (group->status.Failure()) { return group->status; }
        totalBytes += group->totalBytes.load(std::memory_order_relaxed);
        return Status::OK();
    };

    std::unique_ptr<MemoryPool> pool0;
    std::unique_ptr<MemoryPool> pool1;
    try {
        pool0 = load_pool_cache_.Acquire(srcSize, batchShardCount);
        pool1 = load_pool_cache_.Acquire(srcSize, batchShardCount);
    } catch (const std::exception&) {
        markFailure(Status::OutOfMemory());
        ct.waiter->Done();
        return;
    }
    LoadBatch currentBatch;
    LoadBatch nextBatch;
    auto cleanup = [&]() {
        releaseBatch(currentBatch);
        releaseBatch(nextBatch);
    };

    Status s = Status::OK();
    size_t nextBegin = 0;
    size_t totalDecompBytes = 0;
    do {
        nextBegin = prepareBatch(currentBatch, *pool0, 0);
        if (currentBatch.end == currentBatch.begin && !desc.empty()) {
            s = Status::NoSpace();
            break;
        }
        auto result = backend_->Load(std::move(currentBatch.backendDesc));
        s = waitBackendLoad(result, currentBatch);
        if (s.Failure()) { break; }

        while (nextBegin < desc.size()) {
            nextBegin = prepareBatch(nextBatch, *pool1, nextBegin);
            if (nextBatch.end == nextBatch.begin) {
                s = Status::NoSpace();
                break;
            }
            auto nextResult = backend_->Load(std::move(nextBatch.backendDesc));
            s = decompressBatch(currentBatch, totalDecompBytes);
            if (s.Failure()) { break; }
            s = waitBackendLoad(nextResult, nextBatch);
            if (s.Failure()) { break; }
            std::swap(currentBatch, nextBatch);
            std::swap(pool0, pool1);
        }
        if (s.Failure()) { break; }
        s = decompressBatch(currentBatch, totalDecompBytes);
    } while (0);
    if (s.Failure()) {
        cleanup();
        load_pool_cache_.Release(std::move(pool0));
        load_pool_cache_.Release(std::move(pool1));
        markFailure(s);
        ct.waiter->Done();
        return;
    }
    load_pool_cache_.Release(std::move(pool0));
    load_pool_cache_.Release(std::move(pool1));

    UC_DEBUG("COMPRESS LOAD END. Total decompressed bytes: {}", totalDecompBytes);
#else
    // to posix load
    /* 原路径：直接调用 PosixStore */
    backend_->Load(std::move(ct.task->desc));
    UC_DEBUG("COMPRESS LOAD END.");
#endif
    UC_DEBUG("COMPRESS LOAD END, task: {}", ct.task->id);
    ct.waiter->Done();
}

void CompressorAction::Decompress_OneShard(DecompressShardTask& task)
{
    auto done = [&]() { task.group->waiter.Done(); };
    if (task.dst == nullptr || task.src == nullptr) {
        std::lock_guard<std::mutex> lock(task.group->mutex);
        if (task.group->status.Success()) {
            task.group->status = Status::InvalidParam("invalid decompress src/dst");
        }
        done();
        return;
    }

    UC_DEBUG("Decompress start... src {} dst {} shard {}", task.src, task.dst, task.shardIndex);
    size_t decompBytes = 0;
    if (task.ratio == R1) {
        memcpy(task.dst, task.src, task.srcSize);
        decompBytes = task.srcSize;
    } else {
        decompBytes = HUF_decompress_float_fixRatio(task.dst, task.dstSize, task.src, task.srcSize, NULL);
        if (HUF_isError(decompBytes)) {
            std::lock_guard<std::mutex> lock(task.group->mutex);
            if (task.group->status.Success()) {
                task.group->status = Status::Error(
                    fmt::format("failed to decompress shard({}): {}", task.shardIndex,
                                HUF_getErrorName(decompBytes)));
            }
            done();
            return;
        }
    }
    if (decompBytes != task.dstSize) {
        std::lock_guard<std::mutex> lock(task.group->mutex);
        if (task.group->status.Success()) {
            task.group->status = Status::Error(
                fmt::format("unexpected decompressed size {} for shard({}), expect {}",
                            decompBytes, task.shardIndex, task.dstSize));
        }
        done();
        return;
    }
    task.group->totalBytes.fetch_add(decompBytes, std::memory_order_relaxed);
    UC_DEBUG("Decompress end... shard {} decompBytes {}", task.shardIndex, decompBytes);
    done();
}


void CompressorAction::Compress_Dump(CompressTask& ct)
{
#ifdef USE_C_COMPRESS
    UC_DEBUG("COMPRESS DUMP STARTING...");
    const auto& desc = ct.task->desc;
    if (desc.empty()) {
        UC_DEBUG("COMPRESS DUMP desc is empty...");
        return;
    }

    size_t srcSize = shardSize_;
    size_t compBufSize = srcSize + 4096;              // 压缩后缓冲区的可用大小
    
    Detail::TaskDesc backendDesc;
    backendDesc.brief = ct.task->desc.brief;
    std::vector<void*> blockToFree;
    std::unique_ptr<MemoryPool> dump_memoryPool_;
    try {
        dump_memoryPool_ = dump_pool_cache_.Acquire(compBufSize, ct.task->desc.size());
    } catch (const std::exception&) {
        UC_ERROR("Out of memory: failed to allocate {} B", shardSize_ * ct.task->desc.size());
        if (failureSet_) { failureSet_->Insert(ct.task->id); }
        ct.waiter->Done();
        return;
    }

    for (const UC::Detail::Shard& s : desc) {
        UC_DEBUG("Task id: {} Shard index: {}  Compress start...", ct.task->id, s.index);

        uint8_t* compBuf = static_cast<uint8_t*>(dump_memoryPool_->allocate());
        if (compBuf == nullptr) {
            UC_ERROR("No scratch buffer for dump task {} shard {}", ct.task->id, s.index);
            dump_memoryPool_->deallocate(blockToFree);
            dump_pool_cache_.Release(std::move(dump_memoryPool_));
            if (failureSet_) { failureSet_->Insert(ct.task->id); }
            ct.waiter->Done();
            return;
        }
        uint16_t* src = static_cast<uint16_t*>(s.addrs[0]);

        size_t compBytes = 0;
        if (ratio == R1) {
            memcpy(compBuf, src, srcSize);
            compBytes = srcSize;
        } else {
            compBytes = HUF_compress_float_fixRatio (compBuf, compBufSize, src, srcSize, ratio, DT_BF16);
        }
        
        std::vector<void*> _addrs{static_cast<void*>(compBuf)};

        backendDesc.push_back(Detail::Shard {
            s.owner,
            s.index,
            _addrs
        });

        UC_DEBUG("Shard index: {} compress end...  compBytes is {}", s.index, compBytes);
        blockToFree.push_back(static_cast<void*>(compBuf));
    }

    auto res = backend_->Dump(std::move(backendDesc));

    if (!res) {
        UC_ERROR("Failed({}) to submit dump task({}) to backend.", res.Error(), ct.task->id);
        dump_memoryPool_->deallocate(blockToFree);
        dump_pool_cache_.Release(std::move(dump_memoryPool_));
        if (failureSet_) { failureSet_->Insert(ct.task->id); }
        ct.waiter->Done();
        return;
    }

    if (!blockToFree.empty() && res.Value() > 0) {
        auto s = backend_->Wait(res.Value());
        if (s.Failure()) {
            UC_ERROR("Failed({}) to wait dump backend task({}).", s, ct.task->id);
            if (failureSet_) { failureSet_->Insert(ct.task->id); }
        }
        dump_memoryPool_->deallocate(blockToFree);
    }
    dump_pool_cache_.Release(std::move(dump_memoryPool_));

    UC_DEBUG("COMPRESS DUMP END.");
#else
    // to posix dump
    const auto n = ct.task->desc.size();
    if (n > 0) 
    {
        backend_->Dump(std::move(ct.task->desc));
    }

    UC_DEBUG("COMPRESS DUMP END.");
#endif
    ct.waiter->Done();
}

}
