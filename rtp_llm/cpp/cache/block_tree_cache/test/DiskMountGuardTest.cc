#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskMountGuard.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

std::string makeMountDir(const std::string& name) {
    const std::string dir = ::testing::TempDir() + "/disk_mount_guard_test_" + name;
    ::mkdir(dir.c_str(), 0755);
    return dir;
}

void touch(const std::string& path) {
    std::ofstream ofs(path);
    ofs << "stale";
}

bool exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

}  // namespace

TEST(DiskMountGuardTest, InitCreatesWorkDirAndTakesLock) {
    const std::string       mount = makeMountDir("basic");
    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(mount));
    EXPECT_EQ(guard.mountPath(), mount);
    EXPECT_EQ(guard.workDir(), mount + "/rtp_llm_disk_kv");
    EXPECT_TRUE(exists(guard.workDir()));
    EXPECT_TRUE(exists(guard.workDir() + "/.lock"));
    EXPECT_FALSE(guard.debugString().empty());
}

TEST(DiskMountGuardTest, InitRemovesStaleFilesButKeepsLockAndOthers) {
    const std::string mount    = makeMountDir("cleanup");
    const std::string work_dir = mount + "/rtp_llm_disk_kv";
    ::mkdir(work_dir.c_str(), 0755);
    touch(work_dir + "/disk_block_pool_block_tree_disk_r0_l0.bin");  // stale backing -> removed
    touch(work_dir + "/scratch.tmp");                                // stale temp    -> removed
    touch(work_dir + "/keep_me.txt");                                // unrelated     -> kept

    BlockTreeDiskMountGuard guard;
    ASSERT_TRUE(guard.init(mount));
    EXPECT_FALSE(exists(work_dir + "/disk_block_pool_block_tree_disk_r0_l0.bin"));
    EXPECT_FALSE(exists(work_dir + "/scratch.tmp"));
    EXPECT_TRUE(exists(work_dir + "/keep_me.txt"));
}

TEST(DiskMountGuardTest, InitFailsWhenMountMissing) {
    BlockTreeDiskMountGuard guard;
    EXPECT_FALSE(guard.init(::testing::TempDir() + "/disk_mount_guard_test_does_not_exist_xyz"));
}

TEST(DiskMountGuardTest, SecondGuardOnSameMountFailsToLock) {
    const std::string       mount = makeMountDir("double_lock");
    BlockTreeDiskMountGuard first;
    ASSERT_TRUE(first.init(mount));
    BlockTreeDiskMountGuard second;
    EXPECT_FALSE(second.init(mount));  // first still holds LOCK_EX
}

}  // namespace rtp_llm
