#include <atomic>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include "compress/cc/compressor.h"
#include "compress/cc/compress_lib/huf.h"
#include "detail/mock_store.h"
#include "detail/path_base.h"
#include "detail/types_helper.h"
#include "posix/cc/posix_store.h"

namespace {

using MockStore = UC::Test::Detail::MockStore;

UC::Detail::Dictionary MakeCompressConfig(const std::shared_ptr<UC::StoreV1>& backend,
                                          size_t shardSize, int32_t compressRatio = 23)
{
    UC::Detail::Dictionary config;
    config.Set("store_backend", backend);
    config.Set("unique_id", std::string("compress_ut"));
    config.SetNumber("device_id", 0);
    config.SetNumber("tensor_size", shardSize);
    config.SetNumber("shard_size", shardSize);
    config.SetNumber("block_size", shardSize);
    config.SetNumber("layer_size", 1);
    config.SetNumber("compress_ratio", compressRatio);
    config.SetNumber("data_type", 0);
    config.SetNumber("timeout_ms", 3000);
    config.SetNumber("stream_number", 4);
    return config;
}

UC::Detail::TaskHandle NextHandle()
{
    static std::atomic<size_t> id{1};
    return id.fetch_add(1, std::memory_order_relaxed);
}

void* MakeAlignedBuffer(size_t size)
{
    void* ptr = nullptr;
    auto ret = posix_memalign(&ptr, 4096, size);
    EXPECT_EQ(ret, 0);
    return ptr;
}

void FillBf16Pattern(void* buffer, size_t bytes)
{
    auto* p = static_cast<uint16_t*>(buffer);
    const size_t n = bytes / sizeof(uint16_t);
    for (size_t i = 0; i < n; ++i) { p[i] = static_cast<uint16_t>((i / 64) % 13); }
}

UC::Detail::TaskDesc MakeSingleShardTaskDesc(void* addr, const char* brief)
{
    UC::Detail::TaskDesc desc;
    desc.brief = brief;
    desc.push_back(UC::Detail::Shard{
        UC::Test::Detail::TypesHelper::MakeBlockId("compress-block-0001"),
        0,
        {addr},
    });
    return desc;
}

}  // namespace

class UCCompressorTest : public ::testing::Test {
protected:
    static constexpr size_t kShardSize = 131072;
};

TEST_F(UCCompressorTest, LoadReturnsErrorWhenBackendSubmitFailed)
{
    using namespace testing;
    auto backend = std::make_shared<MockStore>();
    EXPECT_CALL(*backend, Load).WillOnce(Return(UC::Status::Error("backend submit failed")));
    EXPECT_CALL(*backend, Wait).Times(0);

    UC::Compressor::Compressor store;
    ASSERT_EQ(store.Setup(MakeCompressConfig(backend, kShardSize)), UC::Status::OK());

    auto* dst = MakeAlignedBuffer(kShardSize);
    auto handle = store.Load(MakeSingleShardTaskDesc(dst, "load-submit-failed"));
    ASSERT_TRUE(handle.HasValue());
    auto status = store.Wait(handle.Value());
    ASSERT_TRUE(status.Failure());
    free(dst);
}

TEST_F(UCCompressorTest, LoadReturnsErrorWhenBackendWaitFailed)
{
    using namespace testing;
    auto backend = std::make_shared<MockStore>();
    EXPECT_CALL(*backend, Load).WillOnce(Invoke([](UC::Detail::TaskDesc task)
                                                    -> UC::Expected<UC::Detail::TaskHandle> {
        EXPECT_EQ(task.size(), size_t{1});
        return UC::Expected<UC::Detail::TaskHandle>(NextHandle());
    }));
    EXPECT_CALL(*backend, Wait).WillOnce(Return(UC::Status::Error("backend wait failed")));

    UC::Compressor::Compressor store;
    ASSERT_EQ(store.Setup(MakeCompressConfig(backend, kShardSize)), UC::Status::OK());

    auto* dst = MakeAlignedBuffer(kShardSize);
    auto handle = store.Load(MakeSingleShardTaskDesc(dst, "load-wait-failed"));
    ASSERT_TRUE(handle.HasValue());
    auto status = store.Wait(handle.Value());
    ASSERT_TRUE(status.Failure());
    free(dst);
}

TEST_F(UCCompressorTest, LoadReturnsErrorWhenCompressedDataIsCorrupted)
{
    using namespace testing;
    auto backend = std::make_shared<MockStore>();
    EXPECT_CALL(*backend, Load).WillOnce(Invoke([](UC::Detail::TaskDesc task)
                                                    -> UC::Expected<UC::Detail::TaskHandle> {
        EXPECT_EQ(task.size(), size_t{1});
        auto* buf = static_cast<uint8_t*>(task[0].addrs[0]);
        std::memset(buf, 0xFF, 90112);
        return UC::Expected<UC::Detail::TaskHandle>(NextHandle());
    }));
    EXPECT_CALL(*backend, Wait).WillOnce(Return(UC::Status::OK()));

    UC::Compressor::Compressor store;
    ASSERT_EQ(store.Setup(MakeCompressConfig(backend, kShardSize)), UC::Status::OK());

    auto* dst = MakeAlignedBuffer(kShardSize);
    auto handle = store.Load(MakeSingleShardTaskDesc(dst, "load-corrupted"));
    ASSERT_TRUE(handle.HasValue());
    auto status = store.Wait(handle.Value());
    ASSERT_TRUE(status.Failure());
    free(dst);
}

class UCCompressorPosixRoundTripTest : public UC::Test::Detail::PathBase {
protected:
    static constexpr size_t kShardSize = 131072;

    UC::Detail::Dictionary MakePosixConfig(const std::string& path, size_t shardSize,
                                           int32_t deviceId = 0) const
    {
        UC::Detail::Dictionary config;
        config.Set("storage_backends", std::vector<std::string>{path});
        config.SetNumber("device_id", deviceId);
        config.SetNumber("tensor_size", shardSize);
        config.SetNumber("shard_size", shardSize);
        config.SetNumber("block_size", shardSize);
        config.SetNumber("timeout_ms", 3000);
        config.SetNumber("posix_data_trans_concurrency", 2);
        config.SetNumber("posix_lookup_concurrency", 2);
        return config;
    }
};

TEST_F(UCCompressorPosixRoundTripTest, DumpThenLoadViaPosixBackend)
{
    UC::PosixStore::PosixStore backend;
    const size_t compressedShardSize = ((kShardSize * size_t{23}) / 32) / 4096 * 4096;
    auto backendConfig = MakePosixConfig(Path(), compressedShardSize);
    ASSERT_EQ(backend.Setup(backendConfig), UC::Status::OK());

    auto backendPtr = std::shared_ptr<UC::StoreV1>(&backend, [](auto) {});
    UC::Compressor::Compressor store;
    ASSERT_EQ(store.Setup(MakeCompressConfig(backendPtr, kShardSize, 23)), UC::Status::OK());

    void* src = MakeAlignedBuffer(kShardSize);
    void* dst = MakeAlignedBuffer(kShardSize);
    std::memset(dst, 0, kShardSize);
    FillBf16Pattern(src, kShardSize);

    auto dumpHandle = store.Dump(MakeSingleShardTaskDesc(src, "roundtrip-dump"));
    ASSERT_TRUE(dumpHandle.HasValue());
    ASSERT_EQ(store.Wait(dumpHandle.Value()), UC::Status::OK());

    auto loadHandle = store.Load(MakeSingleShardTaskDesc(dst, "roundtrip-load"));
    ASSERT_TRUE(loadHandle.HasValue());
    ASSERT_EQ(store.Wait(loadHandle.Value()), UC::Status::OK());
    ASSERT_EQ(std::memcmp(src, dst, kShardSize), 0);

    free(src);
    free(dst);
}
