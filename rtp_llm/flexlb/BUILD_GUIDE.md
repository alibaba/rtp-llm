# FlexLB Build Guide

This guide provides comprehensive instructions for building FlexLB using both Maven and Bazel build systems.

## Prerequisites

### System Requirements
- Linux, macOS, or Windows with WSL2
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Software Requirements
- Java 8 or later (Java 21 recommended for Linux)
- Maven 3.6+ or Bazel 7.0+
- Git
- Docker (optional, for containerized builds)

### Platform-Specific Setup

#### Linux
```bash
# Install Java (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install openjdk-8-jdk maven

# Install Java (RHEL/CentOS/AlmaLinux)
sudo yum install java-1.8.0-openjdk-devel maven

# Install Bazel
curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o bazelisk
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazel
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install openjdk@8 maven bazelisk
```

#### Windows (WSL2)
Follow the Linux instructions within your WSL2 environment.

## Building with Maven

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-org/flexlb.git
cd flexlb

# Build all modules
./mvnw clean package

# Build without tests
./mvnw clean package -DskipTests

# Build specific module
./mvnw clean package -pl flexlb-api -am
```

### Advanced Maven Options
```bash
# Parallel build (uses all CPU cores)
./mvnw clean package -T 1C

# Build with specific Java version
JAVA_HOME=/path/to/java8 ./mvnw clean package

# Build with custom settings
./mvnw clean package -s custom-settings.xml

# Generate site documentation
./mvnw clean site
```

### Maven Profiles
```bash
# Production build
./mvnw clean package -Pprod

# Development build with debugging
./mvnw clean package -Pdev

# Build without internal dependencies
./mvnw clean package -P!internal
```

## Building with Bazel

### Quick Start
```bash
# Build all modules
bazel build //rtp_llm/flexlb:all

# Build specific module
bazel build //rtp_llm/flexlb:flexlb-api

# Run tests
bazel test //rtp_llm/flexlb:test

# Clean build
bazel clean
```

### Advanced Bazel Options
```bash
# Release build
bazel build -c opt //rtp_llm/flexlb:all

# Debug build
bazel build -c dbg //rtp_llm/flexlb:all

# Build with custom Java runtime
bazel build --java_runtime_version=11 //rtp_llm/flexlb:all

# CI build
bazel build --config=ci //rtp_llm/flexlb:all
```

### Bazel Query Examples
```bash
# Show all targets
bazel query //rtp_llm/flexlb:*

# Show dependencies
bazel query "deps(//rtp_llm/flexlb:flexlb-api)"

# Show reverse dependencies
bazel query "rdeps(//rtp_llm/flexlb:all, //rtp_llm/flexlb:flexlb-common)"
```

## Docker Builds

### Building Docker Images
```bash
# Build using Maven
docker build -f docker/Dockerfile.maven -t flexlb:latest .

# Build using Bazel
docker build -f docker/Dockerfile.bazel -t flexlb:latest .

# Multi-stage build (recommended)
docker build -f docker/Dockerfile -t flexlb:latest .
```

### Running in Docker
```bash
# Run with default configuration
docker run -p 7002:7002 -p 8804:8804 flexlb:latest

# Run with custom configuration
docker run -p 7002:7002 -p 8804:8804 \
  -e WHALE_MASTER_CONFIG=/config/master.json \
  -v $(pwd)/config:/config \
  flexlb:latest
```

## Troubleshooting

### Common Maven Issues

**Problem**: Dependencies cannot be downloaded
```bash
# Solution: Check proxy settings
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
./mvnw clean package
```

**Problem**: Out of memory error
```bash
# Solution: Increase heap size
export MAVEN_OPTS="-Xmx4g -XX:MaxPermSize=512m"
./mvnw clean package
```

### Common Bazel Issues

**Problem**: Bazel cannot find Java
```bash
# Solution: Set JAVA_HOME
export JAVA_HOME=/path/to/java
bazel build //rtp_llm/flexlb:all
```

**Problem**: Build cache issues
```bash
# Solution: Clean cache
bazel clean --expunge
bazel shutdown
```

## IDE Integration

### IntelliJ IDEA
1. Install Bazel plugin from marketplace
2. Import project as Maven project
3. Configure SDK to Java 8
4. Enable annotation processing

### Visual Studio Code
1. Install Java Extension Pack
2. Install Bazel extension
3. Configure java.home in settings
4. Use Maven or Bazel view for building

### Eclipse
1. Install m2e plugin
2. Import as Maven project
3. Configure Java 8 compiler
4. Set up Bazel builder (optional)

## Continuous Integration

### GitHub Actions Example
```yaml
name: Build FlexLB
on: [push, pull_request]

jobs:
  build-maven:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-java@v3
        with:
          java-version: '8'
      - run: ./mvnw clean package

  build-bazel:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: bazelbuild/setup-bazelisk@v2
      - run: bazel build //rtp_llm/flexlb:all
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    stages {
        stage('Build with Maven') {
            steps {
                sh './mvnw clean package'
            }
        }
        stage('Build with Bazel') {
            steps {
                sh 'bazel build //rtp_llm/flexlb:all'
            }
        }
    }
}
```

## Performance Tips

### Maven Performance
- Use Maven daemon: `mvnd` instead of `mvn`
- Enable parallel builds: `-T 1C`
- Skip unnecessary plugins: `-Dspotbugs.skip=true`
- Use local repository manager (Nexus/Artifactory)

### Bazel Performance
- Enable remote caching
- Use persistent workers
- Configure proper resource limits
- Use build farm for distributed builds

## Release Process

### Creating a Release
```bash
# Update version (Maven)
./mvnw versions:set -DnewVersion=1.0.0
./mvnw versions:commit

# Build release artifacts
./mvnw clean deploy -Prelease

# Create release package (Bazel)
bazel build --config=release //rtp_llm/flexlb:prepare-release
```

### Publishing Artifacts
```bash
# Deploy to Maven Central
./mvnw clean deploy -Prelease -DskipTests

# Upload to GitHub Releases
gh release create v1.0.0 bazel-bin/rtp_llm/flexlb/*.jar
```

## Contributing

Please ensure all builds pass before submitting PR:
```bash
# Run full validation
./mvnw clean verify
bazel test //rtp_llm/flexlb:test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.