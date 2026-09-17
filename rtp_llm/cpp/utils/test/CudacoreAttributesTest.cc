#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include <dirent.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

constexpr int kAttribEnableOnException = 1;
constexpr int kAttribTriggerHost       = 2;
constexpr int kAttribLightweight       = 3;
constexpr int kAttribEnableUserTrigger = 4;
constexpr int kAttribFile              = 5;
constexpr int kAttribPipe              = 6;
constexpr int kAttribGenerationFlags   = 7;

constexpr int kFakeSuccess            = 0;
constexpr int kFakeInvalidValue       = 1;
constexpr int kFakeInvalidContext     = 201;
constexpr int kFakeNotSupported       = 801;

struct FakeDriver {
    std::map<int, bool>        bools_global;
    std::map<int, bool>        bools_context;
    std::map<int, std::string> strings_global;
    std::map<int, std::string> strings_context;
    uint64_t                   generation_flags{0};
    int                        bool_code{kFakeSuccess};
    int                        string_code{kFakeSuccess};
    int                        context_code{kFakeSuccess};
    int                        global_calls{0};
    int                        context_calls{0};
};

FakeDriver* g_fake = nullptr;

int fakeGet(bool global_scope, int attrib, void* value, size_t* size) {
    if (g_fake == nullptr || size == nullptr) {
        return kFakeNotSupported;
    }
    if (global_scope) {
        ++g_fake->global_calls;
    } else {
        ++g_fake->context_calls;
        if (g_fake->context_code != kFakeSuccess) {
            return g_fake->context_code;
        }
    }
    if (attrib >= kAttribEnableOnException && attrib <= kAttribEnableUserTrigger) {
        if (g_fake->bool_code != kFakeSuccess) {
            return g_fake->bool_code;
        }
        const auto& store = global_scope ? g_fake->bools_global : g_fake->bools_context;
        const auto  entry = store.find(attrib);
        if (entry == store.end()) {
            return kFakeNotSupported;
        }
        if (value == nullptr) {
            *size = sizeof(bool);
            return kFakeSuccess;
        }
        if (*size < sizeof(bool)) {
            return kFakeInvalidValue;
        }
        *reinterpret_cast<bool*>(value) = entry->second;
        *size                           = sizeof(bool);
        return kFakeSuccess;
    }
    if (attrib == kAttribFile || attrib == kAttribPipe) {
        if (g_fake->string_code != kFakeSuccess) {
            return g_fake->string_code;
        }
        const auto& store = global_scope ? g_fake->strings_global : g_fake->strings_context;
        const auto  entry = store.find(attrib);
        if (entry == store.end()) {
            return kFakeNotSupported;
        }
        const std::string& text = entry->second;
        if (value == nullptr) {
            *size = text.size() + 1;
            return kFakeSuccess;
        }
        if (*size < text.size() + 1) {
            *size = text.size() + 1;
            return kFakeInvalidValue;
        }
        std::memcpy(value, text.c_str(), text.size() + 1);
        *size = text.size() + 1;
        return kFakeSuccess;
    }
    if (attrib == kAttribGenerationFlags) {
        if (value == nullptr) {
            *size = sizeof(uint64_t);
            return kFakeSuccess;
        }
        *reinterpret_cast<uint64_t*>(value) = g_fake->generation_flags;
        return kFakeSuccess;
    }
    return kFakeInvalidValue;
}

int fakeGetGlobal(int attrib, void* value, size_t* size) {
    return fakeGet(/*global_scope=*/true, attrib, value, size);
}

int fakeGetContext(int attrib, void* value, size_t* size) {
    return fakeGet(/*global_scope=*/false, attrib, value, size);
}

class CudacoreAttributesTest: public ::testing::Test {
protected:
    void SetUp() override {
        fake_ = FakeDriver{};
        g_fake = &fake_;
        cudacore_test::setDiagnosticsDirOverride(::testing::TempDir() + "/cudacore_attr_test");
    }

    void TearDown() override {
        g_fake = nullptr;
        cudacore_test::resetAttributeApi();
        cudacore_test::resetDiagnosticsDirOverride();
    }

    void installFake() {
        cudacore_test::AttributeApi api;
        api.get_global = &fakeGetGlobal;
        api.get        = &fakeGetContext;
        cudacore_test::installAttributeApi(api);
    }

    FakeDriver fake_;
};

TEST_F(CudacoreAttributesTest, ReportsEffectiveGlobalAndContextValues) {
    fake_.bools_global[kAttribEnableOnException] = true;
    fake_.bools_global[kAttribEnableUserTrigger] = true;
    fake_.bools_global[kAttribTriggerHost]       = false;
    fake_.bools_global[kAttribLightweight]       = false;
    fake_.bools_context[kAttribEnableOnException] = true;
    fake_.strings_global[kAttribFile]            = "prefill_cudacore.%h.%p.%t";
    fake_.strings_global[kAttribPipe]            = "/tmp/cudacore.pipe";
    fake_.strings_context[kAttribFile]           = "prefill_cudacore.%h.%p.%t";
    fake_.generation_flags                       = 0x23;  // skip flags used by the incident
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_EQ(snapshot.symbols_source, "stub");
    EXPECT_EQ(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Ok);
    EXPECT_TRUE(snapshot.global.enable_on_exception.value);
    EXPECT_EQ(snapshot.global.enable_user_trigger.status, CudacoreAttributeStatus::Ok);
    EXPECT_TRUE(snapshot.global.enable_user_trigger.value);
    EXPECT_EQ(snapshot.global.file.status, CudacoreAttributeStatus::Ok);
    EXPECT_EQ(snapshot.global.file.value, "prefill_cudacore.%h.%p.%t");
    EXPECT_FALSE(snapshot.global.file.truncated);
    EXPECT_EQ(snapshot.global.pipe.value, "/tmp/cudacore.pipe");
    EXPECT_EQ(snapshot.global.generation_flags.value, 0x23u);
    EXPECT_TRUE(snapshot.context.context_valid);
    EXPECT_EQ(snapshot.context.enable_on_exception.status, CudacoreAttributeStatus::Ok);
    EXPECT_EQ(snapshot.context.file.value, "prefill_cudacore.%h.%p.%t");
}

TEST_F(CudacoreAttributesTest, MissingSymbolsReportUnsupportedNeverFalse) {
    fake_.bools_global[kAttribEnableOnException] = false;  // never reached: symbols absent
    cudacore_test::AttributeApi api;                       // installed, but no getters at all
    cudacore_test::installAttributeApi(api);

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_EQ(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Unsupported);
    EXPECT_EQ(snapshot.global.enable_user_trigger.status, CudacoreAttributeStatus::Unsupported);
    EXPECT_EQ(snapshot.global.file.status, CudacoreAttributeStatus::Unsupported);
    EXPECT_EQ(snapshot.global.pipe.status, CudacoreAttributeStatus::Unsupported);
    EXPECT_EQ(snapshot.global.generation_flags.status, CudacoreAttributeStatus::Unsupported);
    EXPECT_EQ(snapshot.context.enable_on_exception.status, CudacoreAttributeStatus::Unsupported);
    // Unknown must not be reported as a valid false value.
    EXPECT_NE(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Ok);
}

TEST_F(CudacoreAttributesTest, AttributeLevelUnsupportedIsNotDisguisedAsFalse) {
    fake_.bools_global[kAttribEnableOnException] = false;
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_EQ(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Ok);
    EXPECT_FALSE(snapshot.global.enable_on_exception.value);
    // user trigger was never configured in the fake: not supported, not "false".
    EXPECT_EQ(snapshot.global.enable_user_trigger.status, CudacoreAttributeStatus::Unsupported);
}

TEST_F(CudacoreAttributesTest, ContextQueryFailureIsReportedAsUnavailable) {
    fake_.bools_global[kAttribEnableOnException] = true;
    fake_.context_code                          = kFakeInvalidContext;
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_EQ(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Ok);
    EXPECT_EQ(snapshot.context.enable_on_exception.status, CudacoreAttributeStatus::Unavailable);
    EXPECT_EQ(snapshot.context.file.status, CudacoreAttributeStatus::Unavailable);
    EXPECT_NE(snapshot.context.enable_on_exception.status, CudacoreAttributeStatus::Ok);
}

TEST_F(CudacoreAttributesTest, InvalidContextArgumentSkipsContextScopeWithoutCallingTheDriver) {
    fake_.bools_global[kAttribEnableOnException] = true;
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/false);

    EXPECT_FALSE(snapshot.context.context_valid);
    EXPECT_EQ(snapshot.context.enable_on_exception.status, CudacoreAttributeStatus::Skipped);
    EXPECT_EQ(snapshot.context.file.status, CudacoreAttributeStatus::Skipped);
    EXPECT_EQ(fake_.context_calls, 0);
    EXPECT_GT(fake_.global_calls, 0);
}

TEST_F(CudacoreAttributesTest, OversizedAttributeIsNeverFabricated) {
    fake_.strings_global[kAttribFile] = std::string(8000, 'x');
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_NE(snapshot.global.file.status, CudacoreAttributeStatus::Ok);
    EXPECT_TRUE(snapshot.global.file.value.empty());
}

TEST_F(CudacoreAttributesTest, DriverErrorCodeIsRecorded) {
    fake_.bool_code = 999;  // CUDA_ERROR_UNKNOWN
    installFake();

    const CudacoreAttributeSnapshot snapshot = queryCudacoreAttributes(/*context_valid=*/true);

    EXPECT_EQ(snapshot.global.enable_on_exception.status, CudacoreAttributeStatus::Error);
    EXPECT_EQ(snapshot.global.enable_on_exception.api_code, 999);
}

TEST_F(CudacoreAttributesTest, AuditIsReadOnlyAndDoesNotTouchEnvironment) {
    fake_.bools_global[kAttribEnableOnException] = false;
    fake_.strings_global[kAttribFile]            = "core.%h.%p";
    installFake();

    const char* before_file = std::getenv("CUDA_COREDUMP_FILE");
    const char* before_flag = std::getenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION");
    const std::string file_value = before_file == nullptr ? std::string("<unset>") : std::string(before_file);
    const std::string flag_value = before_flag == nullptr ? std::string("<unset>") : std::string(before_flag);

    (void)queryCudacoreAttributes(/*context_valid=*/true);

    const char* after_file = std::getenv("CUDA_COREDUMP_FILE");
    const char* after_flag = std::getenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION");
    EXPECT_EQ(file_value, after_file == nullptr ? std::string("<unset>") : std::string(after_file));
    EXPECT_EQ(flag_value, after_flag == nullptr ? std::string("<unset>") : std::string(after_flag));
    // The module exposes no setter at all; the query only observed the values.
    EXPECT_EQ(fake_.context_calls > 0, true);
}

TEST_F(CudacoreAttributesTest, StartupAuditWritesManifestWithIdentity) {
    fake_.bools_global[kAttribEnableOnException] = true;
    fake_.bools_global[kAttribEnableUserTrigger] = true;
    fake_.strings_global[kAttribFile]            = "core.%h.%p.%t";
    fake_.strings_global[kAttribPipe]            = "corepipe";
    installFake();
    cudacore_test::resetIncidentState();
    setCudacoreProcessIdentity(/*rank=*/3, /*device_ordinal=*/1);

    auditCudacoreAttributesAtStartup();

    const std::string dir = cudacoreDiagnosticsDir();
    ASSERT_FALSE(dir.empty());

    std::string         manifest_text;
    std::string         manifest_name;
    DIR*                directory = ::opendir(dir.c_str());
    ASSERT_NE(directory, nullptr);
    while (struct dirent* entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name.rfind("cudacore_attributes.", 0) != 0) {
            continue;
        }
        manifest_name = name;
        std::string   content;
        const std::string path = dir + "/" + name;
        FILE*             file = ::fopen(path.c_str(), "r");
        if (file != nullptr) {
            char  buffer[4096] = {};
            const size_t read = ::fread(buffer, 1, sizeof(buffer) - 1, file);
            content.assign(buffer, read);
            ::fclose(file);
        }
        manifest_text = content;
    }
    ::closedir(directory);
    ASSERT_FALSE(manifest_name.empty());
    EXPECT_NE(manifest_text.find("startup_coredump_attributes"), std::string::npos);
    EXPECT_NE(manifest_text.find("\"rank\":3"), std::string::npos);
    EXPECT_NE(manifest_text.find("core.%h.%p.%t"), std::string::npos);
}

TEST_F(CudacoreAttributesTest, ReadinessReportsWritableDirectoryAndFifoPipe) {
    installFake();
    const std::string dir       = ::testing::TempDir() + "/cudacore_readiness_ok";
    const std::string pipe_path = dir + "/cudacore_pipe";
    ASSERT_EQ(::mkdir(dir.c_str(), 0755), 0);
    ASSERT_EQ(::mkfifo(pipe_path.c_str(), 0644), 0);
    ASSERT_EQ(::setenv("CUDA_COREDUMP_PIPE", pipe_path.c_str(), 1), 0);
    cudacore_test::setDiagnosticsDirOverride(dir);

    const CudacoreReadiness readiness = checkCudacoreReadiness();

    EXPECT_TRUE(readiness.dir_ok);
    EXPECT_TRUE(readiness.dir_writable);
    EXPECT_EQ(readiness.dir, dir);
    EXPECT_TRUE(readiness.pipe_configured);
    EXPECT_TRUE(readiness.pipe_exists);
    EXPECT_TRUE(readiness.pipe_is_fifo);

    ::unsetenv("CUDA_COREDUMP_PIPE");
}

TEST_F(CudacoreAttributesTest, ReadinessReportsNonFifoPipeAndUnwritableDirectory) {
    installFake();
    const std::string dir       = ::testing::TempDir() + "/cudacore_readiness_bad";
    const std::string pipe_path = dir + "/not_a_fifo";
    ASSERT_EQ(::mkdir(dir.c_str(), 0755), 0);
    {
        FILE* file = ::fopen(pipe_path.c_str(), "w");
        ASSERT_NE(file, nullptr);
        ::fclose(file);
    }
    ASSERT_EQ(::setenv("CUDA_COREDUMP_PIPE", pipe_path.c_str(), 1), 0);
    // A regular file cannot host a directory: the write probe must fail.
    cudacore_test::setDiagnosticsDirOverride(pipe_path + "/subdir");

    const CudacoreReadiness readiness = checkCudacoreReadiness();

    EXPECT_FALSE(readiness.dir_ok);
    EXPECT_FALSE(readiness.dir_writable);
    EXPECT_TRUE(readiness.pipe_configured);
    EXPECT_TRUE(readiness.pipe_exists);
    EXPECT_FALSE(readiness.pipe_is_fifo);

    ::unsetenv("CUDA_COREDUMP_PIPE");
}

TEST_F(CudacoreAttributesTest, StartupAuditManifestCarriesReadiness) {
    fake_.bools_global[kAttribEnableOnException] = true;
    installFake();
    const std::string dir = ::testing::TempDir() + "/cudacore_readiness_manifest";
    ASSERT_EQ(::mkdir(dir.c_str(), 0755), 0);
    cudacore_test::setDiagnosticsDirOverride(dir);

    auditCudacoreAttributesAtStartup();

    std::string manifest_text;
    DIR*        directory = ::opendir(dir.c_str());
    ASSERT_NE(directory, nullptr);
    while (struct dirent* entry = ::readdir(directory)) {
        const std::string name = entry->d_name;
        if (name.rfind("cudacore_attributes.", 0) != 0) {
            continue;
        }
        const std::string path = dir + "/" + name;
        FILE*             file = ::fopen(path.c_str(), "r");
        if (file != nullptr) {
            char  buffer[8192] = {};
            const size_t read  = ::fread(buffer, 1, sizeof(buffer) - 1, file);
            manifest_text.assign(buffer, read);
            ::fclose(file);
        }
    }
    ::closedir(directory);

    EXPECT_NE(manifest_text.find("\"readiness\""), std::string::npos);
    EXPECT_NE(manifest_text.find("\"diagnostics_dir_writable\":true"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
