# Contributing to FlexLB

Thank you for your interest in contributing to FlexLB! We welcome contributions from the community and are pleased to have you join us.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected to see instead**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `master`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

### Prerequisites

- Java 8 or higher
- Maven 3.6+
- Git

### Setup

1. Fork and clone the repository
```bash
git clone https://github.com/your-username/flexlb.git
cd flexlb
```

2. Build the project
```bash
mvn clean package -DskipTests
```

3. Run tests
```bash
mvn test
```

### Project Structure

```
flexlb/
├── flexlb-api/          # Web layer and HTTP endpoints
├── flexlb-common/       # Shared utilities and models
├── flexlb-grpc/         # gRPC client implementation
├── flexlb-sync/         # Core load balancing logic
├── docs/                # Documentation
└── README.md
```

## Coding Standards

### Java Code Style

- Follow standard Java naming conventions
- Use meaningful variable and method names
- Keep methods focused and concise
- Add JavaDoc comments for public APIs
- Use proper exception handling

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
- `feat(api): add new load balancing strategy`
- `fix(grpc): handle connection timeout properly`
- `docs: update installation guide`

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use meaningful test names that describe what is being tested

## Documentation

- Update README.md if needed
- Add inline code comments for complex logic
- Update API documentation for new endpoints
- Add examples for new features

## Review Process

1. All submissions require review before merging
2. We may ask for changes to be made before a PR can be merged
3. We will provide constructive feedback
4. Once approved, a maintainer will merge your PR

## Questions?

If you have questions about contributing, please:

1. Check the existing issues and discussions
2. Create a new issue with the `question` label
3. Join our community discussions

Thank you for contributing to FlexLB!