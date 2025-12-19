# Bazel Integration Guide for FlexLB

This document describes how to use Bazel to build, test, and manage the FlexLB project alongside the existing Maven build system.

## Overview

FlexLB provides Bazel build integration while maintaining full compatibility with the existing Maven build system. This allows teams to choose their preferred build tool or use both in different environments.

## Quick Start

### Building the Project

Build all modules:
```bash
bazel build //rtp_llm/flexlb:all
```

Build a specific module:
```bash
bazel build //rtp_llm/flexlb:flexlb-common
bazel build //rtp_llm/flexlb:flexlb-api
```

### Running Tests

Run all tests:
```bash
bazel test //rtp_llm/flexlb:test
```

Run tests for a specific module:
```bash
bazel test //rtp_llm/flexlb:flexlb-common-test
```

### Generating Documentation

Generate Javadoc:
```bash
bazel build //rtp_llm/flexlb:generate-docs
```

Generate dependency report:
```bash
bazel build //rtp_llm/flexlb:dependency-report
```

## Build Targets

### Module Targets

Each Maven module has a corresponding Bazel target:

- `//rtp_llm/flexlb:flexlb-common` - Core utilities and shared components
- `//rtp_llm/flexlb:flexlb-cache` - Distributed caching layer
- `//rtp_llm/flexlb:flexlb-grpc` - gRPC client implementation
- `//rtp_llm/flexlb:flexlb-sync` - Load balancing and synchronization
- `//rtp_llm/flexlb:flexlb-api` - RESTful API and web layer

### Aggregate Targets

- `//rtp_llm/flexlb:all` - Build all modules
- `//rtp_llm/flexlb:test` - Run all tests
- `//rtp_llm/flexlb:flexlb-dist` - Create distribution package

### CI/CD Targets

- `//rtp_llm/flexlb:ci` - Continuous integration build
- `//rtp_llm/flexlb:ci-test` - Continuous integration tests

## Configuration

### Java Runtime

The build system uses the configured Java runtime at `/opt/taobao/install/ajdk21_21.0.6.0.6` on Linux systems. To override:

```bash
bazel build --java_runtime_version=local_jdk //rtp_llm/flexlb:all
```

### Build Options

Custom configuration flags:

```bash
# Set FlexLB version
bazel build --//rtp_llm/flexlb:flexlb_version=2.0.0 //rtp_llm/flexlb:all

# Set Java version
bazel build --//rtp_llm/flexlb:java_version=11 //rtp_llm/flexlb:all
```

## Integration with Existing Maven Build

The Bazel build system wraps the existing Maven build to ensure compatibility:

1. **Incremental Builds**: Bazel tracks Maven outputs and only rebuilds when sources change
2. **Dependency Management**: Maven handles all dependency resolution
3. **Test Execution**: Tests run through Maven with proper configuration
4. **Artifact Generation**: Output JARs are identical to Maven builds

## Advanced Usage

### Creating a Release Package

```bash
bazel build //rtp_llm/flexlb:prepare-release
```

This creates a tarball with all FlexLB JARs ready for distribution.

### Platform-Specific Builds

The build automatically detects the platform. For cross-platform builds:

```bash
# Build for Linux
bazel build --platforms=@platforms//os:linux //rtp_llm/flexlb:all

# Build for macOS
bazel build --platforms=@platforms//os:osx //rtp_llm/flexlb:all
```

### Parallel Builds

Bazel automatically parallelizes builds. Maven is configured to use 1 thread per CPU core (`-T 1C`).

## Troubleshooting

### Network Access

Maven requires network access to download dependencies. The build rules disable sandboxing to allow this:

```starlark
execution_requirements = {"no-sandbox": "1"}
```

### Memory Issues

If you encounter out-of-memory errors, the build sets Maven memory options:

```
MAVEN_OPTS="-Xmx2g -XX:MaxMetaspaceSize=512m"
```

Adjust these in the BUILD.bazel file if needed.

### Cache Issues

Clear Bazel cache:
```bash
bazel clean --expunge
```

Clear Maven cache:
```bash
rm -rf ~/.m2/repository/org/flexlb
```

## Best Practices

1. **Use Bazel for CI/CD**: Bazel's caching makes it ideal for continuous integration
2. **Use Maven for Development**: Maven's IDE integration may be better for local development
3. **Version Consistency**: Keep the version in BUILD.bazel synchronized with pom.xml
4. **Test Both Systems**: Ensure changes work with both build systems

## Contributing

When modifying the build:

1. Update both Maven (pom.xml) and Bazel (BUILD.bazel) configurations
2. Test with both build systems
3. Document any new targets or configuration options
4. Ensure CI passes with both build systems

## License

The Bazel build integration is provided under the same Apache License 2.0 as the FlexLB project.