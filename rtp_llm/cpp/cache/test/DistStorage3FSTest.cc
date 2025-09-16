#include <filesystem>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "rtp_llm/cpp/cache/DistStorage3FS.h"
#include "rtp_llm/cpp/cache/test/mock/MockDistStorage3FSFile.h"

using namespace std;
using namespace ::testing;
using ::testing::Return;

namespace rtp_llm::threefs {

class DistStorage3FSTest: public ::testing::Test {
protected:
    static DistStorage::Item makeItem(const std::string& key) {
        DistStorage::Item item;
        item.key   = key;
        item.metas = {
            {"BIZ_NAME", "biz"},
            {"LAYOUT_VERSION", "v1"},
            {"CKPT_PATH", "ckpt"},
            {"LORA_CKPT_PATH", "lora"},
            {"SEQ_SIZE_PER_BLOCK", "16"},
            {"DTYPE", "fp16"},
            {"USE_MLA", "0"},
            {"TP_SIZE", "1"},
            {"TP_RANK", "0"},
            {"ITEM_KEY", key},
        };
        return item;
    }

    static shared_ptr<MockDistStorage3FSFile> makeMockFile(const std::string& path = "/tmp/3fs_mock.dat") {
        ThreeFSFileConfig cfg{
            .mountpoint = "/tmp", .filepath = path, .write_thread_pool = nullptr, .metrics_reporter = nullptr};
        ThreeFSIovHandle read_handle, write_handle;
        return make_shared<MockDistStorage3FSFile>(cfg, read_handle, write_handle);
    }
};

// --------------------------- checkInitParams ---------------------------

TEST_F(DistStorage3FSTest, testCheckInitParams_EmptyMountpoint_ReturnFalse) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    DistStorage3FSInitParams params;
    params.mountpoint          = "";
    params.root_dir            = "rtp_llm_test_root";
    params.file_cache_capacity = 100;

    ASSERT_FALSE(storage->checkInitParams(params));
}

TEST_F(DistStorage3FSTest, testCheckInitParams_MountpointNotExists_ReturnFalse) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    const std::string non_exist_mount = std::string("/tmp/3fs_mount_not_exists_") + std::to_string(::getpid());
    ASSERT_FALSE(std::filesystem::exists(non_exist_mount));

    DistStorage3FSInitParams params;
    params.mountpoint          = non_exist_mount;
    params.root_dir            = "rtp_llm_test_root";
    params.file_cache_capacity = 100;

    ASSERT_FALSE(storage->checkInitParams(params));
}

TEST_F(DistStorage3FSTest, testCheckInitParams_EmptyRootDir_ReturnFalse) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    DistStorage3FSInitParams params;
    params.mountpoint          = "/tmp";  // exists
    params.root_dir            = "";
    params.file_cache_capacity = 100;

    ASSERT_FALSE(storage->checkInitParams(params));
}

TEST_F(DistStorage3FSTest, testCheckInitParams_RootDirNotExists_ReturnFalse) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    const std::string root_dir = std::string("rtp_llm_test_root_not_exists_") + std::to_string(::getpid());
    const auto        full     = std::filesystem::path("/tmp") / root_dir;
    if (std::filesystem::exists(full)) {
        std::filesystem::remove_all(full);
    }
    ASSERT_FALSE(std::filesystem::exists(full));

    DistStorage3FSInitParams params;
    params.mountpoint          = "/tmp";
    params.root_dir            = root_dir;
    params.file_cache_capacity = 100;

    ASSERT_FALSE(storage->checkInitParams(params));
}

TEST_F(DistStorage3FSTest, testCheckInitParams_InvalidFileCacheCapacity_ReturnFalse) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    const std::string root_dir = std::string("rtp_llm_test_root_") + std::to_string(::getpid());
    const auto        full     = std::filesystem::path("/tmp") / root_dir;
    std::filesystem::create_directories(full);
    ASSERT_TRUE(std::filesystem::exists(full));

    DistStorage3FSInitParams params;
    params.mountpoint          = "/tmp";
    params.root_dir            = root_dir;
    params.file_cache_capacity = 0;  // invalid

    ASSERT_FALSE(storage->checkInitParams(params));

    std::filesystem::remove_all(full);
}

TEST_F(DistStorage3FSTest, testCheckInitParams_ValidParams_ReturnTrue) {
    auto storage = std::make_shared<DistStorage3FS>(nullptr);

    const std::string root_dir = std::string("rtp_llm_test_root_") + std::to_string(::getpid());
    const auto        full     = std::filesystem::path("/tmp") / root_dir;
    std::filesystem::create_directories(full);
    ASSERT_TRUE(std::filesystem::exists(full));

    DistStorage3FSInitParams params;
    params.mountpoint          = "/tmp";
    params.root_dir            = root_dir;
    params.file_cache_capacity = 10;  // valid

    ASSERT_TRUE(storage->checkInitParams(params));

    std::filesystem::remove_all(full);
}

// --------------------------- lookup ---------------------------

TEST_F(DistStorage3FSTest, testLookup_GetFileFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    DistStorage::Item item;
    item.key = "k_lookup";
    ASSERT_FALSE(storage->lookup(item));
}

TEST_F(DistStorage3FSTest, testLookup_ExistsFalse_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_lookup");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    auto             item = makeItem(key);
    DistStorage::Iov iov1{
        .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, exists()).WillOnce(Return(false));
    ASSERT_FALSE(storage->lookup(item));
}

TEST_F(DistStorage3FSTest, testLookup_ExistsTrue_ReturnTrue) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_lookup_ok");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    auto             item = makeItem(key);
    DistStorage::Iov iov1{
        .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, exists()).WillOnce(Return(true));
    ASSERT_TRUE(storage->lookup(item));
}

// --------------------------- get ---------------------------

TEST_F(DistStorage3FSTest, testGet_GetFileFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    DistStorage::Item item;
    item.key = "k_get";
    ASSERT_FALSE(storage->get(item));
}

TEST_F(DistStorage3FSTest, testGet_ReadFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_read_fail");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    DistStorage::Item item = makeItem(key);
    DistStorage::Iov  iov1{
         .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, read(::testing::_))
        .WillOnce(::testing::Invoke([iov1, iov2](const std::vector<DistStorage::Iov>& iovs) {
            EXPECT_EQ(iovs.size(), 2);
            EXPECT_EQ(iovs[0].len, 4);
            EXPECT_EQ(iovs[0].data, iov1.data);
            EXPECT_FALSE(iovs[0].gpu_mem);
            EXPECT_FALSE(iovs[0].ignore);
            EXPECT_EQ(iovs[1].len, 8);
            EXPECT_EQ(iovs[1].data, iov2.data);
            EXPECT_FALSE(iovs[1].gpu_mem);
            EXPECT_FALSE(iovs[1].ignore);
            return false;
        }));

    ASSERT_FALSE(storage->get(item));
}

TEST_F(DistStorage3FSTest, testGet_ReadSuccess_ReturnTrue) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_read");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    DistStorage::Item item = makeItem(key);
    DistStorage::Iov  iov1{
         .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, read(::testing::_))
        .WillOnce(::testing::Invoke([iov1, iov2](const std::vector<DistStorage::Iov>& iovs) {
            EXPECT_EQ(iovs.size(), 2);
            EXPECT_EQ(iovs[0].len, 4);
            EXPECT_EQ(iovs[0].data, iov1.data);
            EXPECT_FALSE(iovs[0].gpu_mem);
            EXPECT_FALSE(iovs[0].ignore);
            EXPECT_EQ(iovs[1].len, 8);
            EXPECT_EQ(iovs[1].data, iov2.data);
            EXPECT_FALSE(iovs[1].gpu_mem);
            EXPECT_FALSE(iovs[1].ignore);
            return true;
        }));

    ASSERT_TRUE(storage->get(item));
}

// --------------------------- put ---------------------------

TEST_F(DistStorage3FSTest, testPut_GetFileFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    DistStorage::Item item;
    item.key = "k_put";
    ASSERT_FALSE(storage->put(item));
}

TEST_F(DistStorage3FSTest, testPut_WriteSuccess_ReturnTrue) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_put");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    auto             item = makeItem(key);
    DistStorage::Iov iov1{
        .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, write(::testing::_))
        .WillOnce(::testing::Invoke([iov1, iov2](const std::vector<DistStorage::Iov>& iovs) {
            EXPECT_EQ(iovs.size(), 2);
            EXPECT_EQ(iovs[0].len, 4);
            EXPECT_EQ(iovs[0].data, iov1.data);
            EXPECT_FALSE(iovs[0].gpu_mem);
            EXPECT_FALSE(iovs[0].ignore);
            EXPECT_EQ(iovs[1].len, 8);
            EXPECT_EQ(iovs[1].data, iov2.data);
            EXPECT_FALSE(iovs[1].gpu_mem);
            EXPECT_FALSE(iovs[1].ignore);
            return true;
        }));

    ASSERT_TRUE(storage->put(item));
}

TEST_F(DistStorage3FSTest, testPut_WriteFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_put");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    auto             item = makeItem(key);
    DistStorage::Iov iov1{
        .data = shared_ptr<void>(malloc(4), [](void* p) { free(p); }), .len = 4, .gpu_mem = false, .ignore = false};
    DistStorage::Iov iov2{
        .data = shared_ptr<void>(malloc(8), [](void* p) { free(p); }), .len = 8, .gpu_mem = false, .ignore = false};
    item.iovs = {iov1, iov2};

    EXPECT_CALL(*file, write(::testing::_))
        .WillOnce(::testing::Invoke([iov1, iov2](const std::vector<DistStorage::Iov>& iovs) {
            EXPECT_EQ(iovs.size(), 2);
            EXPECT_EQ(iovs[0].len, 4);
            EXPECT_EQ(iovs[0].data, iov1.data);
            EXPECT_FALSE(iovs[0].gpu_mem);
            EXPECT_FALSE(iovs[0].ignore);
            EXPECT_EQ(iovs[1].len, 8);
            EXPECT_EQ(iovs[1].data, iov2.data);
            EXPECT_FALSE(iovs[1].gpu_mem);
            EXPECT_FALSE(iovs[1].ignore);
            return false;
        }));

    ASSERT_FALSE(storage->put(item));
}

// --------------------------- del ---------------------------

TEST_F(DistStorage3FSTest, testDel_GetFileFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    DistStorage::Item item;
    item.key = "k_del";
    ASSERT_FALSE(storage->del(item));
}

TEST_F(DistStorage3FSTest, testDel_DeleteSuccess_ReturnTrue) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_del");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    EXPECT_CALL(*file, del()).WillOnce(Return(true));

    auto item = makeItem(key);
    ASSERT_TRUE(storage->del(item));
}

TEST_F(DistStorage3FSTest, testDel_DeleteFailed_ReturnFalse) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_del_fail");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);

    EXPECT_CALL(*file, del()).WillOnce(Return(false));

    auto item = makeItem(key);
    ASSERT_FALSE(storage->del(item));
}

// --------------------------- getFile ---------------------------

TEST_F(DistStorage3FSTest, testGetFile_FileInCache_ReturnCachedFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    auto key  = std::string("k_cached");
    auto file = makeMockFile();
    storage->putFileToCache(key, file);
    ASSERT_TRUE(storage->file_cache_->contains(key));

    auto item        = makeItem(key);
    auto cached_file = storage->getFile(item, true);
    ASSERT_NE(cached_file, nullptr);
    ASSERT_EQ(cached_file.get(), file.get());
}

TEST_F(DistStorage3FSTest, testGetFile_InvalidMetas_ReturnNull) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);

    DistStorage::Item item;
    item.key         = "k_invalid_metas";
    auto cached_file = storage->getFile(item, true);
    ASSERT_EQ(cached_file, nullptr);
}

TEST_F(DistStorage3FSTest, testGetFile_CacheFileTrue_PutFileToCache_ReturnFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(10000);

    auto item        = makeItem("key");
    auto cached_file = storage->getFile(item, true);
    ASSERT_NE(cached_file, nullptr);
    ASSERT_TRUE(storage->file_cache_->contains("key"));
    auto [found, file] = storage->file_cache_->get("key");
    ASSERT_TRUE(found);
    ASSERT_EQ(cached_file.get(), file.get());
}

TEST_F(DistStorage3FSTest, testGetFile_CacheFileFalse_ReturnFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(10000);

    auto item        = makeItem("key");
    auto cached_file = storage->getFile(item, false);
    ASSERT_NE(cached_file, nullptr);
    ASSERT_FALSE(storage->file_cache_->contains("key"));
}

// --------------------------- getFileFromCache ---------------------------

TEST_F(DistStorage3FSTest, testGetFileFromCache_KeyExists_ReturnFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);
    auto key             = std::string("k2");
    auto file            = makeMockFile();
    storage->putFileToCache(key, file);
    ASSERT_TRUE(storage->file_cache_->contains(key));
    auto cached_file = storage->getFileFromCache(key);
    ASSERT_NE(cached_file, nullptr);
    ASSERT_EQ(cached_file.get(), file.get());
}

TEST_F(DistStorage3FSTest, testGetFileFromCache_KeyNotExists_ReturnNull) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);
    ASSERT_EQ(storage->getFileFromCache("no_such_key"), nullptr);
}

// --------------------------- putFileToCache ---------------------------

TEST_F(DistStorage3FSTest, testPutFileToCache_KeyNotExists_InsertFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);
    auto key             = std::string("k1");
    auto file            = makeMockFile();
    storage->putFileToCache(key, file);
    ASSERT_TRUE(storage->file_cache_->contains(key));
    ASSERT_EQ(storage->getFileFromCache(key).get(), file.get());
}

TEST_F(DistStorage3FSTest, testPutFileToCache_KeyExists_UpdateFile) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);
    auto key             = std::string("k1");
    auto file            = makeMockFile();
    storage->putFileToCache(key, file);
    ASSERT_TRUE(storage->file_cache_->contains(key));
    ASSERT_EQ(storage->getFileFromCache(key).get(), file.get());

    auto file2 = makeMockFile();
    storage->putFileToCache(key, file2);
    ASSERT_TRUE(storage->file_cache_->contains(key));
    ASSERT_EQ(storage->getFileFromCache(key).get(), file2.get());
}

// --------------------------- clearFileCache ---------------------------

TEST_F(DistStorage3FSTest, testClearFileCache_HasEntries) {
    auto storage         = std::make_shared<DistStorage3FS>(nullptr);
    storage->file_cache_ = std::make_shared<LRUCache<string, shared_ptr<DistStorage3FSFile>>>(100);
    storage->putFileToCache("ka", makeMockFile("/tmp/fa"));
    storage->putFileToCache("kb", makeMockFile("/tmp/fb"));

    storage->clearFileCache();
    ASSERT_EQ(storage->getFileFromCache("ka"), nullptr);
    ASSERT_EQ(storage->getFileFromCache("kb"), nullptr);
}

// --------------------------- makeFilepath ---------------------------

TEST_F(DistStorage3FSTest, testMakeFilepath_MissingMeta_ReturnEmpty) {
    auto                               storage = std::make_shared<DistStorage3FS>(nullptr);
    std::map<std::string, std::string> metas;
    ASSERT_EQ(storage->makeFilepath(metas), "");
}

TEST_F(DistStorage3FSTest, testMakeFilepath_ValidMetas_ReturnNotEmpty) {
    auto storage                     = std::make_shared<DistStorage3FS>(nullptr);
    storage->init_params_.mountpoint = "/mountpoint";
    storage->init_params_.root_dir   = "lxq/";
    auto item                        = makeItem("k");
    auto filepath                    = storage->makeFilepath(item.metas);
    ASSERT_EQ(filepath, "/mountpoint/lxq/biz/v1/ckpt/lora/16/fp16/0/1/0/k");
}

}  // namespace rtp_llm::threefs
