#include "rtp_llm/cpp/devices/BufferManager.h"
#include "rtp_llm/cpp/core/TrackerAllocator.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "autil/StackTracer.h"

#include <numeric>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <unistd.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace std;
using ReadLock  = shared_lock<shared_mutex>;
using WriteLock = unique_lock<shared_mutex>;

namespace rtp_llm {

BufferManager::BufferManager(IAllocator*                        device_allocator,
                             IAllocator*                        host_allocator,
                             const ProfilingDebugLoggingConfig& config):
    device_allocator_(device_allocator),
    host_allocator_(host_allocator),
    trace_memory_(config.trace_memory),
    trace_malloc_stack_(config.trace_malloc_stack),
    profiling_debug_logging_config_(config) {
    if (trace_memory_) {
        autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
        DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
    } else if (trace_malloc_stack_) {
        throw std::runtime_error("RTP_LLM_TRACE_MALLOC_STACK must be used with RTP_LLM_TRACE_MALLOC_STACK");
    }
}

BufferManager::~BufferManager() {}

BufferPtr BufferManager::allocate(const BufferParams& params, const BufferHints& hints) {
    try {
        auto buffer = doAllocate(params, hints);
        recordAllcation(params, hints, buffer);
        return buffer;
    } catch (std::exception& e) {
        RTP_LLM_STACKTRACE_LOG_INFO(
            "allocate buffer failed: size %lu, exception: %s, current allocation records:\n%s \n stack traces: ",
            params.sizeInBytes(),
            e.what(),
            printAllocationRecords(device_allocator_).c_str());
        printStackTrace();
        throw;
    }
}

void BufferManager::recycle(Buffer* buffer, IAllocator* allocator) {
    auto data = buffer->data();

    if (recycle_held_) {
        RTP_LLM_LOG_DEBUG("hold recycle buffer: %p [%s][%s]",
                          data,
                          allocation_records_[data].hints.tag.c_str(),
                          buffer->debugString().c_str());
        held_data_.push_back(std::make_pair(data, allocator));
        return;
    }

    recordRecycle(data);
    doRecycle(data, allocator);
}

BufferPtr BufferManager::doAllocate(const BufferParams& params, const BufferHints& hints) {
    const auto allocator = (params.allocation == AllocationType::DEVICE) ? device_allocator_ : host_allocator_;
    const auto shape     = params.dims;
    const auto alloc_bytes =
        accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>()) * getTypeSize(params.type);
    const auto data = params.private_alloc ? allocator->mallocPrivate(alloc_bytes) : allocator->malloc(alloc_bytes);

    const auto deleter = [this, allocator](Buffer* buffer) { this->recycle(buffer, allocator); };
    const auto buffer  = new Buffer(allocator->memoryType(), params.type, shape, data, deleter);
    return BufferPtr(buffer);
}

void BufferManager::doRecycle(void* data, IAllocator* allocator) {
    allocator->free(&data);
}

void BufferManager::setTraceMemory(bool trace_memory) {
    trace_memory_ = trace_memory;
}

void BufferManager::recordAllcation(const BufferParams& params, const BufferHints& hints, const BufferPtr& buffer) {
    if (trace_memory_) {
        auto stack_trace_id = trace_malloc_stack_ ? autil::StackTracer::getInstance()->getTraceId() : 0;
        {
            WriteLock        lock(mutex_);
            AllocationRecord record             = {params.allocation, buffer->sizeBytes(), hints, stack_trace_id};
            allocation_records_[buffer->data()] = record;
        }
        auto       status                = queryStatus();
        const auto device_consumed_bytes = status.device_allocated_bytes + status.device_fragmented_bytes;
        if (device_consumed_bytes > device_max_consumed_bytes_) {
            RTP_LLM_LOG_INFO("Device allocated size + fragmented size reached new maximum %zu bytes (%.2f MB), \n"
                             "previous is %zu bytes (%.2f MB), current stack trace id[%lu]\n  %s",
                             device_consumed_bytes,
                             device_consumed_bytes / (1024.0 * 1024.0),
                             device_max_allocated_bytes_,
                             device_max_allocated_bytes_ / (1024.0 * 1024.0),
                             stack_trace_id,
                             printAllocationRecords(device_allocator_).c_str());
            device_max_consumed_bytes_ = device_consumed_bytes;
        }
        if (status.device_allocated_bytes > device_max_allocated_bytes_) {
            device_max_allocated_bytes_ = status.device_allocated_bytes;
        }
    }
}

void BufferManager::recordRecycle(void* data) {
    if (trace_memory_) {
        {
            WriteLock lock(mutex_);
            allocation_records_.erase(data);
        }
        RTP_LLM_LOG_DEBUG("record recycle: %p [%s]", data, allocation_records_[data].hints.tag.c_str());
    }
}

BufferStatus BufferManager::queryStatus() {
    auto     status = BufferStatus();
    ReadLock lock(mutex_);
    for (const auto& [_, record] : allocation_records_) {
        if (record.allocation_type == AllocationType::HOST) {
            status.host_allocated_bytes += record.bytes;
        } else {
            status.device_allocated_bytes += record.bytes;
        }
    }
    if (auto tracker_allocator_ = dynamic_cast<TrackerAllocator*>(device_allocator_)) {
        const auto tracker_status        = tracker_allocator_->getTrackerStatus();
        status.device_preserved_bytes    = tracker_status.available_size;
        status.device_fragmented_bytes   = tracker_status.fragmented_size;
        status.device_freezed_bytes      = tracker_status.freezed_bytes;
        status.device_max_consumed_bytes = device_max_consumed_bytes_;
    }
    return status;
}

void BufferManager::holdRecycle() {
    if (recycle_held_) {
        throw std::runtime_error("last buffer manager recycle hold is not released");
    }
    recycle_held_ = true;
}

void BufferManager::releaseRecycleHold() {
    if (!recycle_held_) {
        throw std::runtime_error("buffer manager recycle is not held");
    }
    for (const auto& data_pair : held_data_) {
        auto data      = data_pair.first;
        auto allocator = data_pair.second;
        RTP_LLM_LOG_DEBUG("release held buffer data %p [%s]", data, allocation_records_[data].hints.tag.c_str());
        recordRecycle(data);
        doRecycle(data, allocator);
    }
    held_data_.clear();
    recycle_held_ = false;
}

string BufferManager::printAllocationRecords(IAllocator* allocator) {
    if (auto tracker_allocator = dynamic_cast<TrackerAllocator*>(allocator)) {
        auto               tracker_status = tracker_allocator->getTrackerStatus();
        std::ostringstream info;
        std::set<void*>    allocated_ptrs;
        info << "Memory Tracker [" << (int32_t)tracker_allocator->type() << "] Status:\n";
        info << "allocated " << tracker_status.allocated_chunk_count
             << " chunks, size: " << tracker_status.allocated_size << " bytes (" << std::fixed << std::setprecision(2)
             << (tracker_status.allocated_size / (1024.0 * 1024.0)) << " MB)\n"
             << "available " << tracker_status.available_size << " bytes (" << std::fixed << std::setprecision(2)
             << (tracker_status.available_size / (1024.0 * 1024.0)) << " MB), with "
             << tracker_status.fragment_chunk_count << " fragments of size: " << tracker_status.fragmented_size
             << " bytes (" << std::fixed << std::setprecision(2) << (tracker_status.fragmented_size / (1024.0 * 1024.0))
             << " MB)\n";
        {
            ReadLock lock(mutex_);
            for (const auto& [ptr, record] : allocation_records_) {
                allocated_ptrs.insert(ptr);
            }

            // Calculate UTF-8 string display width (CJK characters count as 2, ASCII as 1)
            auto calcDisplayWidth = [](const std::string& str) -> size_t {
                size_t display_width = 0;
                size_t i             = 0;
                while (i < str.length()) {
                    unsigned char c = static_cast<unsigned char>(str[i]);
                    if (c < 0x80) {
                        display_width += 1;
                        i += 1;
                    } else if ((c & 0xE0) == 0xC0) {
                        display_width += 1;
                        i += 2;
                    } else if ((c & 0xF0) == 0xE0) {
                        if (i + 2 < str.length()) {
                            unsigned char c1 = static_cast<unsigned char>(str[i]);
                            unsigned char c2 = static_cast<unsigned char>(str[i + 1]);
                            unsigned char c3 = static_cast<unsigned char>(str[i + 2]);

                            uint32_t codepoint = ((c1 & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

                            if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF)
                                || (codepoint >= 0x3400 && codepoint <= 0x4DBF)
                                || (codepoint >= 0x3000 && codepoint <= 0x303F)
                                || (codepoint >= 0xFF00 && codepoint <= 0xFFEF)) {
                                display_width += 2;
                            } else {
                                display_width += 1;
                            }
                        }
                        i += 3;
                    } else if ((c & 0xF8) == 0xF0) {
                        display_width += 2;
                        i += 4;
                    } else {
                        i += 1;
                    }
                }
                return display_width;
            };

            // Helper struct to store formatted buffer output
            struct BufferOutput {
                std::string                                 first_line;
                std::vector<std::pair<std::string, size_t>> content_lines;  // <line, width>
            };

            // Format a single buffer allocation for output
            auto formatBufferAllocation = [&calcDisplayWidth](void*              ptr,
                                                              size_t             bytes,
                                                              size_t             trace_id,
                                                              const std::string& tag,
                                                              bool               is_tracked_chunk = true,
                                                              bool               is_used = false) -> BufferOutput {
                BufferOutput       output;
                std::ostringstream line_stream;

                if (is_tracked_chunk) {
                    // For tracked chunks, use fixed format with hex size
                    line_stream << ptr << " | " << setw(12) << bytes << " (" << std::fixed << std::setprecision(2)
                                << setw(8) << (bytes / (1024.0 * 1024.0)) << " MB)"
                                << " | " << (is_used ? "USED" : "FREE");
                } else {
                    // For untracked buffers, simpler format
                    line_stream << ptr << " | " << setw(12) << bytes << " (" << std::fixed << std::setprecision(2)
                                << setw(12) << (bytes / (1024.0 * 1024.0)) << " MB)"
                                << " |     ";
                }

                bool isMultiLineTag = (tag.find('\n') != std::string::npos);

                if (isMultiLineTag) {
                    // Multi-line tag: first line shows basic info
                    line_stream << " | " << setw(4) << trace_id << " | [Multi-Line Stack]";
                    std::string first = line_stream.str();
                    output.first_line = first;
                    output.content_lines.push_back({first, calcDisplayWidth(first)});

                    // Parse and store each line of the stack trace
                    std::istringstream iss(tag);
                    std::string        stack_line;
                    while (std::getline(iss, stack_line)) {
                        if (!stack_line.empty()) {
                            size_t width = calcDisplayWidth(stack_line);
                            output.content_lines.push_back({stack_line, width});
                        }
                    }
                } else {
                    // Single-line tag
                    line_stream << " | " << setw(4) << trace_id << " | " << setw(16) << tag;
                    std::string line  = line_stream.str();
                    output.first_line = line;
                    output.content_lines.push_back({line, calcDisplayWidth(line)});
                }

                return output;
            };

            // Collect all output lines and calculate maximum content width
            std::vector<std::pair<std::string, size_t>> all_lines;
            size_t                                      max_content_width = 0;

            for (const auto chunk : tracker_status.chunks) {
                const auto alloc_record = allocation_records_.find(chunk.ptr);
                if (alloc_record != allocation_records_.end()) {
                    allocated_ptrs.erase(chunk.ptr);
                    const std::string& tag = alloc_record->second.hints.tag;

                    // Format this buffer allocation
                    auto output = formatBufferAllocation(chunk.ptr,
                                                         chunk.size,
                                                         alloc_record->second.trace_id,
                                                         tag,
                                                         true,  // is_tracked_chunk
                                                         chunk.used);

                    // Add all lines and update max width
                    for (const auto& [line, width] : output.content_lines) {
                        all_lines.push_back({line, width});
                        max_content_width = std::max(max_content_width, width);
                    }
                } else {
                    // Chunk with no allocation record
                    std::ostringstream line_stream;
                    line_stream << chunk.ptr << " | " << setw(12) << chunk.size << " (" << std::fixed
                                << std::setprecision(2) << setw(8) << (chunk.size / (1024.0 * 1024.0)) << " MB)"
                                << " | " << (chunk.used ? "USED" : "FREE") << " |      |                  ";
                    std::string line  = line_stream.str();
                    size_t      width = calcDisplayWidth(line);
                    all_lines.push_back({line, width});
                    max_content_width = std::max(max_content_width, width);
                }
            }

            // Output table header with proper padding
            size_t total_width = max_content_width + 4;
            info << std::string(total_width, '-') << "\n";
            std::string header_content = "       ADDR |         size (      MB) | AVAIL| TRACE|              TAG";
            size_t      header_padding = max_content_width - calcDisplayWidth(header_content);
            info << "| " << header_content << std::string(header_padding, ' ') << " |\n";
            info << std::string(total_width, '-') << "\n";

            // Output all collected lines with proper padding
            for (const auto& [content, width] : all_lines) {
                size_t padding = max_content_width - width;
                info << "| " << content << std::string(padding, ' ') << " |\n";
            }

            info << std::string(total_width, '-') << "\n";

            // Handle untracked allocated buffers
            if (allocated_ptrs.size()) {
                info << "There are also " << allocated_ptrs.size() << " buffers allocated but not tracked, "
                     << "they are not shown in the list: \n";
                info << std::string(total_width, '-') << "\n";

                for (const auto ptr : allocated_ptrs) {
                    const auto         alloc_record = allocation_records_.find(ptr);
                    const std::string& tag          = alloc_record->second.hints.tag;

                    // Format this buffer allocation using the same function
                    auto output = formatBufferAllocation(ptr,
                                                         alloc_record->second.bytes,
                                                         alloc_record->second.trace_id,
                                                         tag,
                                                         false,  // is_tracked_chunk = false
                                                         false   // is_used (not applicable for untracked)
                    );

                    // Output all lines with proper padding
                    for (const auto& [line, width] : output.content_lines) {
                        size_t padding = (width < max_content_width) ? (max_content_width - width) : 0;
                        info << "| " << line << std::string(padding, ' ') << " |\n";
                    }
                }
                info << std::string(total_width, '-') << "\n";
            }
        }
        return info.str();
    } else {
        RTP_LLM_LOG_WARNING("BufferManager::printAllocationRecords is only effective when using TrackerAllocator!");
        return "";
    }
}

}  // namespace rtp_llm
