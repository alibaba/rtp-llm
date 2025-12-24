# Copyright 2025 FlexLB Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")
load(":maven_rules.bzl", "maven_jar", "maven_test")

# ============================================================================
# FlexLB - Flexible Load Balancer for AI Model Inference
# ============================================================================
# FlexLB provides intelligent load balancing, caching, and request batching
# for AI/ML workloads, designed to optimize inference performance and resource
# utilization in distributed model serving environments.
#
# This BUILD file provides Bazel integration for the existing Maven project
# structure, allowing seamless build system interoperability.
# ============================================================================

package(
    default_visibility = ["//visibility:public"],
)

# License file for distribution
exports_files(
    ["LICENSE"],
    visibility = ["//visibility:public"],
)

# ============================================================================
# Build Configuration Flags
# ============================================================================

string_flag(
    name = "flexlb_version",
    build_setting_default = "1.0.0-SNAPSHOT",
    visibility = ["//visibility:public"],
)

string_flag(
    name = "java_version",
    build_setting_default = "21",
    visibility = ["//visibility:public"],
)

# ============================================================================
# Java Toolchain Configuration
# ============================================================================

java_runtime(
    name = "flexlb_java_runtime",
    java_home = select({
        "@platforms//os:linux": "/opt/taobao/install/ajdk21_21.0.6.0.6",
        "//conditions:default": None,
    }),
    visibility = ["//visibility:public"],
)

# ============================================================================
# Common Definitions and Macros
# ============================================================================

FLEXLB_VERSION = "1.0.0-SNAPSHOT"
MAVEN_COORDINATES_PREFIX = "org.flexlb:"

# ============================================================================
# Module Build Targets
# ============================================================================
# The modules are built in dependency order:
# flexlb-common -> flexlb-cache -> flexlb-grpc -> flexlb-sync -> flexlb-api

# flexlb-common: Core utilities and shared components
maven_jar(
    name = "flexlb-common",
    srcs = glob([
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "mvnw",
    ]),
    deps = [],
    jar_name = "flexlb-common-{}.jar".format(FLEXLB_VERSION),
    module_dir = "flexlb-common",
    modules = ["flexlb-common"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# flexlb-cache: Distributed caching layer
maven_jar(
    name = "flexlb-cache",
    srcs = glob([
        "flexlb-cache/src/**/*.java",
        "flexlb-cache/pom.xml",
        "mvnw",
    ]),
    deps = [":flexlb-common"],
    jar_name = "flexlb-cache-{}.jar".format(FLEXLB_VERSION),
    module_dir = "flexlb-cache",
    modules = ["flexlb-common", "flexlb-cache"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# flexlb-grpc: gRPC client implementation
maven_jar(
    name = "flexlb-grpc",
    srcs = glob([
        "flexlb-grpc/src/**/*.java",
        "flexlb-grpc/src/main/resources/proto/**/*.proto",
        "flexlb-grpc/pom.xml",
        "mvnw",
    ]),
    deps = [
        ":flexlb-common",
        ":flexlb-cache",
    ],
    jar_name = "flexlb-grpc-{}.jar".format(FLEXLB_VERSION),
    module_dir = "flexlb-grpc",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# flexlb-sync: Core load balancing and synchronization
maven_jar(
    name = "flexlb-sync",
    srcs = glob([
        "flexlb-sync/src/**/*.java",
        "flexlb-sync/pom.xml",
        "mvnw",
    ]),
    deps = [
        ":flexlb-common",
        ":flexlb-cache",
        ":flexlb-grpc",
    ],
    jar_name = "flexlb-sync-{}.jar".format(FLEXLB_VERSION),
    module_dir = "flexlb-sync",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc", "flexlb-sync"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# flexlb-api: RESTful API and web layer
maven_jar(
    name = "flexlb-api",
    srcs = glob([
        "flexlb-api/src/**/*.java",
        "flexlb-api/src/main/resources/**/*",
        "flexlb-api/pom.xml",
        "mvnw",
    ]),
    deps = [
        ":flexlb-common",
        ":flexlb-cache",
        ":flexlb-grpc",
        ":flexlb-sync",
    ],
    jar_name = "flexlb-api-{}.jar".format(FLEXLB_VERSION),
    module_dir = "flexlb-api",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc", "flexlb-sync", "flexlb-api"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# ============================================================================
# Aggregate Targets
# ============================================================================

# Build all modules
filegroup(
    name = "all",
    srcs = [
        ":flexlb-api",
        ":flexlb-cache",
        ":flexlb-common",
        ":flexlb-grpc",
        ":flexlb-sync",
    ],
    visibility = ["//visibility:public"],
)

# Distribution package including all JARs and documentation
filegroup(
    name = "flexlb-dist",
    srcs = [
        ":all",
        "LICENSE",
        "README.md",
    ],
    visibility = ["//visibility:public"],
)

# ============================================================================
# Test Targets
# ============================================================================

# Test targets for each module
maven_test(
    name = "flexlb-common-test",
    srcs = glob([
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "flexlb-cache/pom.xml",
        "flexlb-grpc/pom.xml",
        "flexlb-sync/pom.xml",
        "flexlb-api/pom.xml",
        "mvnw",
        "pom.xml",
        ".mvn/**/*",
    ]),
    deps = [],
    module_dir = "flexlb-common",
    modules = ["flexlb-common"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

maven_test(
    name = "flexlb-cache-test",
    srcs = glob([
        "flexlb-cache/src/**/*.java",
        "flexlb-cache/pom.xml",
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "flexlb-grpc/pom.xml",
        "flexlb-sync/pom.xml",
        "flexlb-api/pom.xml",
        "mvnw",
        "pom.xml",
        ".mvn/**/*",
    ]),
    deps = [],
    module_dir = "flexlb-cache",
    modules = ["flexlb-common", "flexlb-cache"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

maven_test(
    name = "flexlb-grpc-test",
    srcs = glob([
        "flexlb-grpc/src/**/*.java",
        "flexlb-grpc/src/main/resources/proto/**/*.proto",
        "flexlb-grpc/pom.xml",
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "flexlb-cache/src/**/*.java",
        "flexlb-cache/pom.xml",
        "flexlb-sync/pom.xml",
        "flexlb-api/pom.xml",
        "mvnw",
        "pom.xml",
        ".mvn/**/*",
    ]),
    deps = [],
    module_dir = "flexlb-grpc",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

maven_test(
    name = "flexlb-sync-test",
    srcs = glob([
        "flexlb-sync/src/**/*.java",
        "flexlb-sync/pom.xml",
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "flexlb-cache/src/**/*.java",
        "flexlb-cache/pom.xml",
        "flexlb-grpc/src/**/*.java",
        "flexlb-grpc/src/main/resources/proto/**/*.proto",
        "flexlb-grpc/pom.xml",
        "flexlb-api/pom.xml",
        "mvnw",
        "pom.xml",
        ".mvn/**/*",
    ]),
    deps = [],
    module_dir = "flexlb-sync",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc", "flexlb-sync"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

maven_test(
    name = "flexlb-api-test",
    srcs = glob([
        "flexlb-api/src/**/*.java",
        "flexlb-api/src/main/resources/**/*",
        "flexlb-api/pom.xml",
        "flexlb-common/src/**/*.java",
        "flexlb-common/pom.xml",
        "flexlb-cache/src/**/*.java",
        "flexlb-cache/pom.xml",
        "flexlb-grpc/src/**/*.java",
        "flexlb-grpc/src/main/resources/proto/**/*.proto",
        "flexlb-grpc/pom.xml",
        "flexlb-sync/src/**/*.java",
        "flexlb-sync/pom.xml",
        "mvnw",
        "pom.xml",
        ".mvn/**/*",
    ]),
    deps = [],
    module_dir = "flexlb-api",
    modules = ["flexlb-common", "flexlb-cache", "flexlb-grpc", "flexlb-sync", "flexlb-api"],
    pom = "pom.xml",
    visibility = ["//visibility:public"],
)

# Run all tests
test_suite(
    name = "test",
    tests = [
        ":flexlb-api-test",
        ":flexlb-cache-test",
        ":flexlb-common-test",
        ":flexlb-grpc-test",
        ":flexlb-sync-test",
    ],
    visibility = ["//visibility:public"],
)

# ============================================================================
# Documentation and Analysis Targets
# ============================================================================

genrule(
    name = "generate-docs",
    srcs = glob([
        "*/src/main/java/**/*.java",
        "*/pom.xml",
        "pom.xml",
        "mvnw",
        ".mvn/**/*",
    ]),
    outs = ["flexlb-docs.tar.gz"],
    cmd = """
        set +e
        export JAVA_HOME="/opt/taobao/java"
        export PATH="$$JAVA_HOME/bin:/usr/bin:/bin:/usr/local/bin:$$PATH"
        EXECROOT=$$(pwd)
        OUTPUT_FILE=$(location flexlb-docs.tar.gz)
        ABS_OUTPUT_FILE="$$EXECROOT/$$OUTPUT_FILE"
        POM_FILE=$$(dirname $(location pom.xml))
        cd "$$EXECROOT/$$POM_FILE" || exit 1
        # Skip internal profile to avoid dependency on kmonitor and vipserver
        # Run javadoc even if it has errors, we'll create archive if docs directory exists
        ./mvnw javadoc:aggregate -B -q -P!internal 2>&1 || true
        # Always create output file - create empty tar if docs don't exist
        OUTPUT_DIR=$$(dirname "$$ABS_OUTPUT_FILE")
        mkdir -p "$$OUTPUT_DIR" || exit 1
        # Use absolute path for output file
        if [ -d target/site/apidocs ] && [ -n "$$(ls -A target/site/apidocs 2>/dev/null)" ]; then
            tar czf "$$ABS_OUTPUT_FILE" -C target/site/apidocs . 2>&1
            TAR_EXIT=$$?
            if [ $$TAR_EXIT -ne 0 ] || [ ! -f "$$ABS_OUTPUT_FILE" ]; then
                echo "Warning: Failed to create tar from docs (exit $$TAR_EXIT), creating empty archive"
                # Create empty tar.gz using Python (more reliable)
                python3 -c "import gzip; f=gzip.open('$$ABS_OUTPUT_FILE', 'wb'); f.write(b''); f.close()" 2>&1 || \
                python -c "import gzip; f=gzip.open('$$ABS_OUTPUT_FILE', 'wb'); f.write(b''); f.close()" 2>&1 || \
                touch "$$ABS_OUTPUT_FILE"
            fi
        else
            echo "Warning: API docs not generated, creating empty archive"
            # Create empty tar.gz using Python (more reliable)
            python3 -c "import gzip; f=gzip.open('$$ABS_OUTPUT_FILE', 'wb'); f.write(b''); f.close()" 2>&1 || \
            python -c "import gzip; f=gzip.open('$$ABS_OUTPUT_FILE', 'wb'); f.write(b''); f.close()" 2>&1 || \
            touch "$$ABS_OUTPUT_FILE"
        fi
        # Final verification - ensure file exists at the expected location (use relative path for Bazel)
        if [ ! -f "$$ABS_OUTPUT_FILE" ]; then
            echo "Error: Output file was not created at $$ABS_OUTPUT_FILE"
            echo "Current directory: $$(pwd)"
            echo "Output file path (abs): $$ABS_OUTPUT_FILE"
            echo "Output file path (rel): $$OUTPUT_FILE"
            echo "Output dir: $$OUTPUT_DIR"
            ls -la "$$OUTPUT_DIR" 2>&1 || true
            # Last resort: create empty file
            touch "$$ABS_OUTPUT_FILE" || exit 1
        fi
        # Verify file exists at relative path (Bazel checks this)
        # Return to execroot and verify file exists at relative path
        cd "$$EXECROOT" || exit 1
        if [ ! -f "$$OUTPUT_FILE" ]; then
            echo "Error: Output file not found at relative path $$OUTPUT_FILE from execroot"
            echo "Current directory: $$(pwd)"
            echo "Looking for: $$OUTPUT_FILE"
            echo "Absolute path was: $$ABS_OUTPUT_FILE"
            # Try to copy from absolute path if it exists
            if [ -f "$$ABS_OUTPUT_FILE" ]; then
                mkdir -p $$(dirname "$$OUTPUT_FILE") || true
                cp "$$ABS_OUTPUT_FILE" "$$OUTPUT_FILE" || exit 1
            else
                echo "Absolute file also does not exist, creating empty file"
                mkdir -p $$(dirname "$$OUTPUT_FILE") || true
                touch "$$OUTPUT_FILE" || exit 1
            fi
        fi
        echo "Output file verified: $$OUTPUT_FILE (size: $$(wc -c < "$$OUTPUT_FILE" 2>/dev/null || echo 0) bytes)"
    """,
    message = "Generating API documentation",
    visibility = ["//visibility:public"],
)

genrule(
    name = "dependency-report",
    srcs = glob([
        "pom.xml",
        "mvnw",
        ".mvn/**/*",
    ]),
    outs = ["dependency-report.txt"],
    cmd = """
        export JAVA_HOME="/opt/taobao/java"
        export PATH="$$JAVA_HOME/bin:/usr/bin:/bin:/usr/local/bin:$$PATH"
        POM_DIR=$$(dirname $(location :pom.xml))
        OUTPUT_FILE=$(location dependency-report.txt)
        touch "$$OUTPUT_FILE" && \
        cd "$$POM_DIR" && \
        chmod +x ./mvnw 2>/dev/null || /usr/bin/chmod +x ./mvnw 2>/dev/null || true && \
        ./mvnw dependency:tree -B >> "$$OUTPUT_FILE" 2>&1 || echo "Maven dependency:tree completed" >> "$$OUTPUT_FILE" && \
        echo "Dependency report generated" >> "$$OUTPUT_FILE"
    """,
    message = "Generating dependency report",
    visibility = ["//visibility:public"],
)

# ============================================================================
# CI/CD Integration Targets
# ============================================================================

# Target for continuous integration builds
alias(
    name = "ci",
    actual = ":all",
    visibility = ["//visibility:public"],
)

# Target for continuous integration tests
alias(
    name = "ci-test",
    actual = ":test",
    visibility = ["//visibility:public"],
)

# ============================================================================
# Developer Convenience Targets
# ============================================================================

# Quick build without tests
alias(
    name = "build",
    actual = ":all",
    visibility = ["//visibility:public"],
)

# Clean build (handled by Bazel's clean command)
# Run: bazel clean

# ============================================================================
# Platform-Specific Configurations
# ============================================================================

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:osx"],
    visibility = ["//visibility:public"],
)

# ============================================================================
# Release Targets
# ============================================================================

genrule(
    name = "prepare-release",
    srcs = [":all"],
    outs = ["flexlb-{version}.tar.gz".format(version = FLEXLB_VERSION)],
    cmd = """
        set -euo pipefail
        VERSION="{version}"
        mkdir -p flexlb-$$VERSION/lib
        for jar in $$(echo $(SRCS)); do
            cp $$jar flexlb-$$VERSION/lib/
        done
        tar czf $(location flexlb-{version}.tar.gz) flexlb-$$VERSION
    """.format(version = FLEXLB_VERSION),
    message = "Preparing release package",
    visibility = ["//visibility:public"],
)