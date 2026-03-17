#ifndef UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H
#define UNIFIEDCACHE_COMPRESSOR_CC_ACTION_H

#include <unistd.h> 
#include <atomic>
#include <mutex>
#include "global_config.h"
#include "template/hashset.h"
#include "trans_task.h"
#include "thread/latch.h"
#include "ucmstore_v1.h"
#include "thread/thread_pool.h"
#include "compress_lib/huf.h"  // HUF_compress_float_fixRatio, HUF_decompress_float_fixRatio
#include "memory_pool.h"

namespace UC::Compressor {

#define USE_C_COMPRESS

class CompressorAction {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

private:
    std::shared_ptr<StoreV1> backend_{nullptr};
    TaskIdSet* failureSet_{nullptr};
    size_t shardSize_{0};
    FixedRatio ratio{R145};
    DataType dataType{DT_INVALID};
    
    struct CompressTask {
        std::shared_ptr<TransTask> task;
        std::shared_ptr<Latch> waiter;
    };
    struct DecompressGroup {
        Latch waiter;
        std::atomic<size_t> totalBytes{0};
        std::mutex mutex;
        Status status{Status::OK()};
    };
    struct DecompressShardTask {
        Detail::TaskHandle owner{0};
        size_t shardIndex{0};
        void* src{nullptr};
        void* dst{nullptr};
        size_t srcSize{0};
        size_t dstSize{0};
        FixedRatio ratio{R145};
        std::shared_ptr<DecompressGroup> group;
    };
    class MemoryPoolCache {
    public:
        std::unique_ptr<MemoryPool> Acquire(size_t blockSize, size_t poolSize);
        void Release(std::unique_ptr<MemoryPool> pool);

    private:
        std::mutex mutex_;
        std::vector<std::unique_ptr<MemoryPool>> pools_;
    };
    ThreadPool<CompressTask> dump_pool_;
    ThreadPool<CompressTask> load_pool_;
    ThreadPool<DecompressShardTask> decompress_pool_;
    MemoryPoolCache load_pool_cache_;
    MemoryPoolCache dump_pool_cache_;

public:
    ~CompressorAction();
    Status Setup(const Config& config, TaskIdSet* failureSet);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void Compress_Load(CompressTask& ios);
    void Compress_Dump(CompressTask& ios);
    void Decompress_OneShard(DecompressShardTask& task);

};

}

#endif
