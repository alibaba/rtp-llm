# Python Stack Trace and Memory Allocation Tracking

## Overview

RTP-LLM provides a powerful memory tracking and Python stack trace feature for debugging and analyzing memory allocation behavior at runtime. This functionality captures Python-layer call stacks and associates them with underlying C++ memory allocations, helping developers quickly identify the source of memory allocations.

## Key Features

- **Python Stack Tracing**: Automatically captures Python call stack during memory allocation
- **Path Simplification**: Simplifies file path display for better readability
- **Beautified Output**: Uses Unicode box-drawing characters for clear stack presentation
- **Memory Tracking Integration**: Deeply integrated with BufferManager's memory tracking functionality

## Quick Start

### When to Use

This feature is particularly useful when:

1. **Runtime Memory Doesn't Match Expectations**: When actual memory usage significantly differs from expected values
   - GPU memory usage exceeds model size estimates
   - Unexpected memory peaks during inference
   - Memory keeps growing over time (potential memory leak)
   - Out-of-memory (OOM) errors occur unexpectedly

2. **Memory Leak Debugging**: When you notice memory continuously growing

### Configuration

#### Environment Variables

Enable memory tracking and Python stack tracing through environment variables:

```bash
# Enable PyTorch allocator Python stack tracing
 `export ENABLE_TORCH_ALLOC_PROFILE=1` or `--enable_torch_alloc_profile 1`
```

## Output Examples

### Memory Allocation Report Format

At warmup phase, when device memory usage reaches a new peak, the system automatically outputs a detailed memory allocation report.

```
[2026-01-12 17:57:57.570162] [INFO] [RANK 1] Device allocated size + fragmented size reached new maximum 50416016 bytes (48.08 MB),
previous is 16865680 bytes (16.08 MB), current stack trace id[0]
  Memory Tracker [1] Status:
allocated 11 chunks, size: 83970944 bytes (80.08 MB)
available 81551012992 bytes (77773.11 MB), with 0 fragments of size: 0 bytes (0.00 MB)
```

### Memory Block List

```
--------------------------------------------------------------------------------------------------------
|        ADDR |         size (      MB) | AVAIL| TRACE|              TAG                               |
--------------------------------------------------------------------------------------------------------
| 0x7f0d64000000 |     33554432 (   32.00 MB) | USED |      |                                          |
| 0x7f0d66000000 |        24576 (    0.02 MB) | USED |    0 |      exp_log_cnt                         |
| 0x7f0d66006000 |          384 (    0.00 MB) | USED |    0 |     phy_gpu_load                         |
```

### Python Stack Trace Display

For memory allocations containing Python stacks, multi-line formatted stack information is displayed:

```
| 0x7f0d66016380 |     16775168 (   16.00 MB) | USED |    0 | [Multi-Line Stack]                       |
|   ┌─ Python Stack Trace ─────────────────────────────────────────                                    |
|   │    github-opensource/rtp_llm/models_py/model_desc/generic_moe.py:308 in forward                  |
|   │    └─ inputs_embeds = self.embed_tokens(input_ids)                                               |
|   │    github-opensource/rtp_llm/models_py/modules/base/common/embedding.py:30 in forward            |
|   │    └─ output = torch.empty(                                                                      |
|   └──────────────────────────────────────────────────────────────                                    |
```

### Complete Example Output

```
| 0x7f0d67015b80 |     33550336 (   32.00 MB) | USED |    0 | [Multi-Line Stack]                       |
|   ┌─ Python Stack Trace ─────────────────────────────────────────                                    |
|   │    github-opensource/rtp_llm/models_py/model_desc/generic_moe.py:308 in forward                  |
|   │    └─ inputs_embeds = self.embed_tokens(input_ids)                                               |
|   │    github-opensource/rtp_llm/models_py/modules/base/common/embedding.py:36 in forward            |
|   │    └─ output = all_gather(output, group=Group.TP)                                                |
|   │    github-opensource/rtp_llm/models_py/distributed/collective_torch.py:341 in all_gather         |
|   │    └─ tensor_list = torch.zeros([world_size * tensor....                                         |
|   └──────────────────────────────────────────────────────────────────────────────────────            |
```

## Design Principles

### Architecture Components

1. **StackTrace.cc/h**: Core stack trace implementation
   - `getPythonStackTrace()`: Get formatted Python stack information
   - `getOriginPythonStack()`: Extract original Python stack frames
   - `reformatPythonStack()`: Format and filter stack frames

2. **TorchCudaAllocator.cc**: PyTorch CUDA memory allocator integration
   - Calls `getPythonStackTrace()` in `allocate()` method
   - Passes stack information as tag to BufferManager

3. **BufferManager.cc**: Memory allocation recording and display
   - Records each memory allocation with associated stack trace
   - Generates detailed memory allocation reports

### Workflow

```
Python Code Call
    ↓
PyTorch Tensor Allocation
    ↓
TorchCudaAllocator::allocate()
    ↓
getPythonStackTrace() ← Capture current Python stack
    ↓
BufferManager::allocate() ← Record allocation info and stack
    ↓
Allocation Record (AllocationRecord)
```

### Stack Filtering Strategy

To improve information effectiveness, the system intelligently filters stack frames:

1. **Prioritize**: Frames containing `model_desc` path (model definition layer)
2. **Auto-filter**: Framework internal calls like `torch/nn/modules/module.py`
3. **Length Limit**: Maximum of 5 frames by default
4. **Head-Tail Preservation**: When exceeding limit, preserve first 2 and last 2 frames

### Path Simplification Rules

- Display from the first directory after `site-packages/`
- Display from the first directory after `RTP-LLM/`
- Example: `/opt/conda/lib/python3.10/site-packages/torch/nn/functional.py` → `torch/nn/functional.py`

## Implementation Details

### Key Data Structures

```cpp
// Memory allocation record
struct AllocationRecord {
    AllocationType allocation_type;  // DEVICE or HOST
    size_t bytes;                    // Allocation size
    BufferHints hints;               // Contains tag (stack trace)
    size_t trace_id;                 // Stack trace ID
};

// Buffer hint information
struct BufferHints {
    std::string tag;                 // Python stack trace string
};
```

### Core Functions

#### getPythonStackTrace()

```cpp
std::string getPythonStackTrace() {
    if (!Py_IsInitialized()) {
        return "Python interpreter not initialized";
    }

    try {
        // 1. Get original stack frames
        std::vector<std::string> frames = getOriginPythonStack();

        // 2. Format and filter stack
        return reformatPythonStack(frames);
    } catch (...) {
        // Exception handling
    }
}
```

#### getOriginPythonStack()

```cpp
static std::vector<std::string> getOriginPythonStack() {
    py::gil_scoped_acquire gil;  // Acquire GIL lock

    PyThreadState* tstate = PyThreadState_GET();
    PyFrameObject* frame = PyThreadState_GetFrame(tstate);

    py::module_ traceback = py::module_::import("traceback");
    py::list stack_list = traceback.attr("extract_stack")(py_frame);

    // Iterate through stack frames, extract filename, line number, function name, code line
    for (auto item : stack_list) {
        // Simplify path
        // Extract code line
        // Format output
    }
}
```

#### reformatPythonStack()

```cpp
static std::string reformatPythonStack(const std::vector<std::string>& frames) {
    // 1. Find model_desc related frames
    // 2. Filter framework internal code
    // 3. Limit frame count (head + tail)
    // 4. Beautify output using Unicode box-drawing characters

    std::stringstream stack_ss;
    stack_ss << "\n  ┌─ Python Stack Trace ─────────\n";
    for (const auto& frame : final_frames) {
        stack_ss << "  │" << frame << "\n";
    }
    stack_ss << "  └──────────────────────────────\n";
}
```

## Advanced Usage

### Adding Custom Filtering Rules

Add in `reformatPythonStack()`:

```cpp
// Add custom critical path priority
if (frames[i].find("my_model/core") != std::string::npos) {
    core_model_index = i;
    break;
}
```

### Integration with Other Memory Allocators

```cpp
class MyAllocator : public IAllocator {
public:
    void* malloc(size_t size) override {
        std::string tag = rtp_llm::getPythonStackTrace();
        // Record allocation info and stack
        return allocate_internal(size, tag);
    }
};
```

### Supporting C++ Stack Traces

```cpp
#include "rtp_llm/cpp/utils/StackTrace.h"

// Get C++ stack trace
std::string cpp_stack = rtp_llm::getStackTrace();

// Combine Python and C++ stack traces
std::string combined =
    rtp_llm::getPythonStackTrace() + "\n" +
    rtp_llm::getStackTrace();
```

## Reference Code

- `rtp_llm/cpp/utils/StackTrace.cc`: Core stack trace implementation
- `rtp_llm/cpp/utils/StackTrace.h`: Interface definition
- `rtp_llm/cpp/core/torch_utils/torch_cuda_allocator.cc`: PyTorch integration
- `rtp_llm/cpp/devices/BufferManager.cc`: Memory tracking integration
