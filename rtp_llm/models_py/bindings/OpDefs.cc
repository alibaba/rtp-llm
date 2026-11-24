#include "OpDefs.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#if USE_PPU
#include <algorithm>
#include <array>
#include <chrono>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <thread>
#include <torch/csrc/autograd/python_variable.h>
#include <unordered_map>
#include <unordered_set>
#endif

namespace torch_ext {

size_t KVCache::getEntriesPerBlock(const std::string& tag) const {
    const auto& group = grouped_layout_.topology().group(tag);
    const auto* spec  = dynamic_cast<const rtp_llm::OpaqueKVCacheSpec*>(group.spec.get());
    if (!spec) {
        return 0;
    }
    size_t entries = spec->entriesPerBlock();
    if (group.policy.group_type == rtp_llm::CacheGroupType::FULL && spec->type == rtp_llm::KVCacheSpecType::OpaqueKV) {
        const auto pages = group.kernelBlocksPerKvBlock();
        RTP_LLM_CHECK_WITH_INFO(
            pages > 0 && entries % pages == 0, "opaque entries must divide kernel pages for tag=%s", tag.c_str());
        entries /= pages;
    }
    return entries;
}

namespace {

#if USE_PPU
// A second slot lets a rank enter the following vote without overwriting a
// result another rank is still reading. No rank can complete two votes ahead.
class PpuScheduleVote {
public:
    PpuScheduleVote(pybind11::buffer mapping, uint32_t rank, uint32_t ranks):
        buffer_(mapping.request(/*writable=*/true)), rank_(rank), ranks_(ranks) {
        TORCH_CHECK(ranks > 1 && rank < ranks, "invalid PPU schedule vote rank");
        TORCH_CHECK(buffer_.ndim == 1 && buffer_.itemsize == 1 && buffer_.strides[0] == 1
                        && size_t(buffer_.size) == size(ranks) && uintptr_t(buffer_.ptr) % alignof(uint64_t) == 0,
                    "invalid PPU schedule vote mapping");
    }

    static size_t size(uint32_t ranks) {
        return 2 * ranks * kSlotBytes;
    }

    void publish(bool reuse, int64_t rows) {
        TORCH_CHECK(!pending_, "PPU schedule vote already published");
        TORCH_CHECK(rows >= 0 && rows <= INT32_MAX, "invalid PPU schedule vote batch");
        const uint32_t epoch = ++epoch_;
        TORCH_CHECK(epoch != 0, "PPU schedule vote generation wrapped");
        const uint64_t prefix = uint64_t(epoch) << 32;
        const uint64_t local = prefix | (uint64_t(rows) << 1) | uint64_t(!reuse);
        __atomic_store_n(slots() + rank_ * kSlotWords, local, __ATOMIC_RELEASE);
        pending_ = true;
    }

    std::pair<bool, bool> collect() {
        TORCH_CHECK(pending_, "PPU schedule vote has no published row");
        auto* const current_slots = slots();
        const uint64_t local = __atomic_load_n(current_slots + rank_ * kSlotWords, __ATOMIC_RELAXED);
        const int64_t rows = uint32_t(local) >> 1;
        const auto start = std::chrono::steady_clock::now();
        const auto spin_until = start + std::chrono::microseconds(30);
        const auto deadline = start + std::chrono::seconds(120);
        bool all_reuse = (local & 1) == 0;
        bool common_bucket = rows > 0;
        for (uint32_t rank = 0; rank < ranks_; ++rank) {
            if (rank == rank_) {
                continue;
            }
            uint64_t value;
            while ((value = __atomic_load_n(current_slots + rank * kSlotWords, __ATOMIC_ACQUIRE)) >> 32 != epoch_) {
                const auto now = std::chrono::steady_clock::now();
                TORCH_CHECK(now < deadline, "PPU schedule vote timed out waiting for rank ", rank,
                            " in generation ", epoch_);
                if (now >= spin_until) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
            all_reuse &= (value & 1) == 0;
            common_bucket &= int64_t(uint32_t(value) >> 1) == rows;
        }
        pending_ = false;
        return {all_reuse, common_bucket};
    }

    std::pair<bool, bool> exchange(bool reuse, int64_t rows) {
        publish(reuse, rows);
        return collect();
    }

private:
    uint64_t* slots() {
        return reinterpret_cast<uint64_t*>(
            static_cast<char*>(buffer_.ptr) + (epoch_ % 2) * ranks_ * kSlotBytes);
    }

    static constexpr size_t kSlotBytes = 64;
    static constexpr size_t kSlotWords = kSlotBytes / sizeof(uint64_t);
    pybind11::buffer_info buffer_;
    const uint32_t rank_;
    const uint32_t ranks_;
    uint32_t epoch_ = 0;
    bool pending_ = false;
};

// One leased publication workspace owns this signal and its copy stream. The
// device's system fence publishes the preceding local KV status D2H to the CPU.
class PpuPublicationReceipt {
public:
    PpuPublicationReceipt():
        word_(torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true))) {
        auto result = cuMemHostGetDevicePointer(&device_word_, word_.data_ptr(), 0);
        TORCH_CHECK(result == CUDA_SUCCESS, "cannot map PPU publication receipt: ", static_cast<int>(result));
    }

    void record() {
        if (pending_) {
            wait();
        }
        const auto stream = c10::cuda::getCurrentCUDAStream();
        const auto next = epoch_ + 1U;
        const auto result = cuStreamWriteValue32(stream.stream(), device_word_, next, 0);
        TORCH_CHECK(result == CUDA_SUCCESS, "cannot enqueue PPU publication receipt: ", static_cast<int>(result));
        epoch_ = next;
        pending_ = stream;
    }

    void wait() {
        if (!pending_) return;
        const auto* value = reinterpret_cast<const uint32_t*>(word_.data_ptr<int32_t>());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (__atomic_load_n(value, __ATOMIC_ACQUIRE) != epoch_) {
            if (std::chrono::steady_clock::now() >= deadline) {
                pending_->synchronize();
                TORCH_CHECK(__atomic_load_n(value, __ATOMIC_ACQUIRE) == epoch_, "missing PPU publication receipt");
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        pending_.reset();
    }

    ~PpuPublicationReceipt() {
        if (pending_) {
            (void)cuStreamSynchronize(pending_->stream());
        }
    }

private:
    torch::Tensor word_;
    CUdeviceptr device_word_{};
    uint32_t epoch_{};
    std::optional<c10::cuda::CUDAStream> pending_;
};

// The PPU graph keeps the addresses observed at capture. Read live module
// bindings on replay, including replacements that keep the same tensor shape.
class GraphWeightStorage {
    struct Entry {
        PyTypeObject*        python_type;
        const void*          address;
        c10::ScalarType      dtype;
        std::vector<int64_t> sizes;
    };
    std::vector<Entry> entries_;
#if PY_VERSION_HEX < 0x030D0000
    struct ModuleWeights {
        std::array<uint64_t, 4> dict_versions{};
        std::vector<pybind11::weakref> tensors;
    };
    std::vector<ModuleWeights> module_weights_;
    pybind11::object names_;
    // CPython assigns a process-wide unique version when an exact dictionary changes.
    // Equal versions keep its values alive. No borrowed tensor is touched after a rebind.
    static uint64_t dictVersion(PyObject* dict) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        return reinterpret_cast<PyDictObject*>(dict)->ma_version_tag;
#pragma GCC diagnostic pop
    }
#endif
    pybind11::str parameters_{"_parameters"}, buffers_{"_buffers"}, children_{"_modules"};
    pybind11::str dict_{"__dict__"}, modules_{"modules"}, named_modules_{"named_modules"};

public:
    explicit GraphWeightStorage(pybind11::list tensors) {
        for (auto value : tensors) {
            TORCH_CHECK(THPVariable_Check(value.ptr()), "Graph weight capture requires tensors");
            const auto& tensor = THPVariable_Unpack(value.ptr());
            entries_.push_back({Py_TYPE(value.ptr()), tensor.data_ptr(), tensor.scalar_type(), tensor.sizes().vec()});
        }
    }

    bool matches(pybind11::list prefix, pybind11::iterable modules, pybind11::list suffix,
                 pybind11::tuple names, pybind11::function plain_lookup) {
        return matchesImpl(prefix, suffix, names, plain_lookup, [&](const auto& check_module) {
            for (auto module : modules) {
                if (!check_module(module, nullptr)) return false;
            }
            return true;
        });
    }

    bool matchesModuleTree(pybind11::list prefix, pybind11::object root, pybind11::list suffix,
                           pybind11::tuple names, pybind11::function standard_traversal) {
        return matchesImpl(prefix, suffix, names, pybind11::function{}, [&](const auto& check_module) {
            std::unordered_set<PyTypeObject*> types;
            std::unordered_set<PyObject*> seen;
            std::vector<pybind11::object> pending{std::move(root)};
            while (!pending.empty()) {
                auto module = std::move(pending.back());
                pending.pop_back();
                auto* type = Py_TYPE(module.ptr());
                if (types.insert(type).second
                    && !standard_traversal(pybind11::handle(reinterpret_cast<PyObject*>(type))).cast<bool>()) {
                    return false;
                }
                const auto attrs = pybind11::getattr(module, dict_);
                if (!PyDict_CheckExact(attrs.ptr()) || PyDict_GetItem(attrs.ptr(), modules_.ptr())
                    || PyDict_GetItem(attrs.ptr(), named_modules_.ptr())) {
                    return false;
                }
                auto* children = PyDict_GetItem(attrs.ptr(), children_.ptr());
                if (!children || !PyDict_CheckExact(children)) return false;
                if (!seen.insert(module.ptr()).second) continue;
                if (!check_module(module, attrs.ptr())) return false;

                const auto first_child = pending.size();
                Py_ssize_t position = 0;
                PyObject* key;
                PyObject* child;
                while (PyDict_Next(children, &position, &key, &child)) {
                    if (child != Py_None) {
                        pending.push_back(pybind11::reinterpret_borrow<pybind11::object>(child));
                    }
                }
                // Match named_modules' depth-first registration order.
                std::reverse(pending.begin() + first_child, pending.end());
            }
            return true;
        });
    }

private:
    template <class Traverse>
    bool matchesImpl(const pybind11::list& prefix, const pybind11::list& suffix, const pybind11::tuple& names,
                     const pybind11::function& plain_lookup, const Traverse& traverse) {
        size_t index = 0;
        auto check = [&](PyObject* object) {
            if (!object || !THPVariable_Check(object)) {
                return true;
            }
            const auto& tensor = THPVariable_Unpack(object);
            if (index == entries_.size()) {
                return false;
            }
            const auto& entry = entries_[index++];
            return Py_TYPE(object) == entry.python_type && tensor.scalar_type() == entry.dtype && tensor.sizes().equals(entry.sizes)
                   && tensor.data_ptr() == entry.address;
        };
        for (auto value : prefix) {
            if (!THPVariable_Check(value.ptr()) || !check(value.ptr())) return false;
        }
        std::unordered_map<PyTypeObject*, bool> direct;
#if PY_VERSION_HEX < 0x030D0000
        if (names_.ptr() != names.ptr()) {
            module_weights_.clear();
            names_ = names;
        }
        size_t position = 0;
#endif
        const auto check_module = [&](pybind11::handle module, PyObject* known_attrs) {
            pybind11::object attrs;
            if (known_attrs) {
                attrs = pybind11::reinterpret_borrow<pybind11::object>(known_attrs);
            } else {
                auto* type = Py_TYPE(module.ptr());
                auto it = direct.find(type);
                if (it == direct.end()) {
                    it = direct.emplace(type, plain_lookup(pybind11::handle(reinterpret_cast<PyObject*>(type))).cast<bool>()).first;
                }
                attrs = it->second ? pybind11::getattr(module, dict_) : pybind11::none();
            }
            auto* dict = attrs.ptr();
            auto* params = PyDict_CheckExact(dict) ? PyDict_GetItem(dict, parameters_.ptr()) : nullptr;
            auto* buffers = PyDict_CheckExact(dict) ? PyDict_GetItem(dict, buffers_.ptr()) : nullptr;
            auto* children = PyDict_CheckExact(dict) ? PyDict_GetItem(dict, children_.ptr()) : nullptr;
#if PY_VERSION_HEX < 0x030D0000
            if (position == module_weights_.size()) module_weights_.emplace_back();
            auto& cached = module_weights_[position++];
#endif
            if (params && buffers && children && PyDict_CheckExact(params)
                && PyDict_CheckExact(buffers) && PyDict_CheckExact(children)) {
#if PY_VERSION_HEX < 0x030D0000
                const std::array<uint64_t, 4> versions{
                    dictVersion(dict), dictVersion(params), dictVersion(buffers), dictVersion(children)};
                if (versions[0] && versions[1] && versions[2] && versions[3] && cached.dict_versions == versions) {
                    for (const auto& tensor : cached.tensors) {
                        auto* value = PyWeakref_GetObject(tensor.ptr());
                        if (!value || value == Py_None || !check(value)) return false;
                    }
                    return true;
                }
                cached.tensors.clear();
                cached.dict_versions = {};
#endif
                for (auto name : names) {
                    auto* value = PyDict_GetItem(dict, name.ptr());
                    if (!value) value = PyDict_GetItem(params, name.ptr());
                    if (!value) value = PyDict_GetItem(buffers, name.ptr());
                    if (!value) value = PyDict_GetItem(children, name.ptr());
                    if (!check(value)) return false;
#if PY_VERSION_HEX < 0x030D0000
                    if (value && THPVariable_Check(value)) cached.tensors.emplace_back(pybind11::handle(value));
#endif
                }
#if PY_VERSION_HEX < 0x030D0000
                cached.dict_versions = versions;
#endif
            } else {
                for (auto name : names) {
                    auto value = pybind11::getattr(module, name, pybind11::none());
                    if (!check(value.ptr())) return false;
                }
            }
            return true;
        };
        if (!traverse(check_module)) return false;
        for (auto value : suffix) {
            if (!THPVariable_Check(value.ptr()) || !check(value.ptr())) return false;
        }
        return index == entries_.size();
    }
};
#endif

void registerCacheGroupType(pybind11::module& m) {
    try {
        auto config_m            = pybind11::module_::import("libth_transformer_config");
        m.attr("CacheGroupType") = config_m.attr("CacheGroupType");
        return;
    } catch (const pybind11::error_already_set& e) {
        PyErr_Clear();
    }

    pybind11::enum_<rtp_llm::CacheGroupType>(m, "CacheGroupType")
        .value("LINEAR", rtp_llm::CacheGroupType::LINEAR)
        .value("FULL", rtp_llm::CacheGroupType::FULL)
        .value("SWA", rtp_llm::CacheGroupType::SWA)
        .export_values();
}

}  // namespace

void registerPyOpDefs(pybind11::module& m) {
    registerCacheGroupType(m);
#if USE_PPU
    pybind11::class_<PpuScheduleVote>(m, "_PpuScheduleVote")
        .def(pybind11::init<pybind11::buffer, uint32_t, uint32_t>())
        .def_static("size", &PpuScheduleVote::size)
        .def("publish", &PpuScheduleVote::publish)
        .def("collect", &PpuScheduleVote::collect, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("exchange", &PpuScheduleVote::exchange, pybind11::call_guard<pybind11::gil_scoped_release>());
    pybind11::class_<PpuPublicationReceipt>(m, "_PpuPublicationReceipt")
        .def(pybind11::init<>())
        .def("record", &PpuPublicationReceipt::record)
        .def("wait", &PpuPublicationReceipt::wait, pybind11::call_guard<pybind11::gil_scoped_release>());
    pybind11::class_<GraphWeightStorage>(m, "GraphWeightStorage")
        .def(pybind11::init<pybind11::list>())
        .def("matches", &GraphWeightStorage::matches)
        .def("matches_module_tree", &GraphWeightStorage::matchesModuleTree);
#endif

    pybind11::class_<LayerKVCache>(m, "LayerKVCache")
        .def(pybind11::init<>())
        .def(pybind11::init([](torch::Tensor    kv_cache_base,
                               int              seq_size_per_block,
                               int              layer_id,
                               std::string      tag,
                               pybind11::object kv_scale_base) {
                 torch::Tensor scale;
                 if (!kv_scale_base.is_none()) {
                     scale = kv_scale_base.cast<torch::Tensor>();
                 }
                 return LayerKVCache(
                     std::move(kv_cache_base), seq_size_per_block, layer_id, std::move(tag), std::move(scale));
             }),
             pybind11::arg("kv_cache_base"),
             pybind11::arg("seq_size_per_block"),
             pybind11::arg("layer_id")      = -1,
             pybind11::arg("tag")           = "default",
             pybind11::arg("kv_scale_base") = pybind11::none())
        .def_readwrite("kv_cache_base", &LayerKVCache::kv_cache_base, "Key/value cache tensor (per-layer view)")
        .def_readwrite("kv_scale_base", &LayerKVCache::kv_scale_base, "Key/value cache scale tensor")
        .def_readonly("seq_size_per_block", &LayerKVCache::seq_size_per_block, "Sequence size per block")
        .def_readonly("layer_id", &LayerKVCache::layer_id, "Global layer id")
        .def_readonly("tag", &LayerKVCache::tag, "Cache group tag");

    pybind11::class_<KVCache>(m, "KVCache")
        .def_property_readonly("group_tags", &KVCache::groupTags, "Cache group tags in topology publication order")
        .def_property_readonly("layer_count", &KVCache::layerCount, "Number of model-local cache layers")
        .def("get_layer_cache",
             static_cast<LayerKVCache (KVCache::*)(int) const>(&KVCache::getLayerCache),
             "Return a per-layer LayerKVCache for the given global layer id")
        .def("get_layer_cache",
             static_cast<LayerKVCache (KVCache::*)(int, const std::string&) const>(&KVCache::getLayerCache),
             "Return a LayerKVCache for the given layer and tag")
        .def("get_layer_cache_groups",
             &KVCache::getLayerCacheGroups,
             "Return every valid LayerKVCache group owned by the layer")
        .def("get_seq_size_per_block",
             &KVCache::getSeqSizePerBlock,
             "Return the physical sequence size per block for a cache tag")
        .def("get_kernel_seq_size_per_block",
             &KVCache::getKernelSeqSizePerBlock,
             "Return the kernel sequence size per block for a cache tag")
        .def("get_entries_per_block",
             &KVCache::getEntriesPerBlock,
             "Return opaque logical entries per layer-cache block, excluding padding; zero for other specs");

    pybind11::class_<PyModelInitResources>(m, "PyModelInitResources")
        .def(pybind11::init<>())
        .def_readonly("kv_cache", &PyModelInitResources::kv_cache, "KV cache for all layers")
        .def_readonly("is_speculative", &PyModelInitResources::is_speculative)
        .def_readonly("is_decode_role", &PyModelInitResources::is_decode_role)
        .def_readonly("max_context_batch_size", &PyModelInitResources::max_context_batch_size);

    pybind11::class_<caffe2::TypeMeta>(m, "TypeMeta").def(pybind11::init<>());

    m.def(
        "get_typemeta",
        [](const torch::Tensor& tensor) { return torch::scalarTypeToTypeMeta(tensor.scalar_type()); },
        "Convert tensor dtype to TypeMeta");

    m.def(
        "get_scalar_type",
        [](caffe2::TypeMeta dtype) { return dtype.toScalarType(); },
        "Convert TypeMeta to scalar type");

    pybind11::class_<PyCacheStoreInputs>(m, "PyCacheStoreInputs").def(pybind11::init<>());

    pybind11::class_<rtp_llm::CacheStoreWriter, std::shared_ptr<rtp_llm::CacheStoreWriter>>(m, "CacheStoreWriter")
        .def("write", &rtp_llm::CacheStoreWriter::write);

    pybind11::class_<rtp_llm::ParamsBase, std::shared_ptr<rtp_llm::ParamsBase>>(m, "ParamsBase")
        .def(pybind11::init<>())
        .def(
            "fill_params",
            [](rtp_llm::ParamsBase& self,
               torch::Tensor        sequence_lengths,
               torch::Tensor        input_lengths,
               torch::Tensor        kv_cache_block_id_host,
               int                  batch_size,
               int                  seq_size_per_block) {
                self.fillParams(
                    sequence_lengths, input_lengths, kv_cache_block_id_host, batch_size, seq_size_per_block);
            },
            pybind11::arg("sequence_lengths"),
            pybind11::arg("input_lengths"),
            pybind11::arg("kv_cache_block_id_host"),
            pybind11::arg("batch_size"),
            pybind11::arg("seq_size_per_block"),
            "Fill parameters for CUDA graph execution");

    pybind11::class_<PyPrefillCudaGaphCopyParams>(m, "PyPrefillCudaGaphCopyParams")
        .def(pybind11::init<>())
        .def_readwrite("cuda_graph_prefill_batch_size", &PyPrefillCudaGaphCopyParams::cuda_graph_prefill_batch_size)
        .def_readwrite("max_seq_len", &PyPrefillCudaGaphCopyParams::max_seq_len)
        .def_readwrite("max_batch_size", &PyPrefillCudaGaphCopyParams::max_batch_size);

    pybind11::class_<PyContextParallelParams>(m, "PyContextParallelParams")
        .def(pybind11::init<>())
        .def_readwrite("prefill_qkv_restore_indice_cpu", &PyContextParallelParams::prefill_qkv_restore_indice_cpu)
        .def_readwrite("prefill_qkv_padding_mask_cpu", &PyContextParallelParams::prefill_qkv_padding_mask_cpu)
        .def_readwrite("prefill_mm_spans", &PyContextParallelParams::prefill_mm_spans)
        .def_readwrite("prefill_cp_padding_lengths", &PyContextParallelParams::prefill_cp_padding_lengths)
        .def_readwrite("prefill_cp_chunk_lengths", &PyContextParallelParams::prefill_cp_chunk_lengths)
        .def_readwrite("prefill_shuffle_indices", &PyContextParallelParams::prefill_shuffle_indices)
        .def_readwrite("prefill_qkv_restore_indice", &PyContextParallelParams::prefill_qkv_restore_indice)
        .def_readwrite("prefill_qkv_padding_mask", &PyContextParallelParams::prefill_qkv_padding_mask)
        .def_readwrite("prefill_actual_input_lengths_cpu", &PyContextParallelParams::prefill_actual_input_lengths_cpu)
        .def_readwrite("prefill_prefix_lengths_cpu", &PyContextParallelParams::prefill_prefix_lengths_cpu);

    pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
        .def(pybind11::init<>())
        .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
        .def_readwrite("is_cuda_graph", &PyAttentionInputs::is_cuda_graph)
        .def_readwrite("is_target_verify", &PyAttentionInputs::is_target_verify)
        .def_readwrite("is_packed_target_verify", &PyAttentionInputs::is_packed_target_verify)
        .def_readwrite("prefix_lengths", &PyAttentionInputs::prefix_lengths)
        .def_readwrite("sequence_lengths", &PyAttentionInputs::sequence_lengths)
        .def_readwrite("input_lengths", &PyAttentionInputs::input_lengths)
        .def_readwrite("kv_cache_kernel_block_id", &PyAttentionInputs::kv_cache_kernel_block_id)
        .def_readwrite("kv_cache_kernel_block_id_device", &PyAttentionInputs::kv_cache_kernel_block_id_device)
        .def_readwrite("kv_cache_block_id", &PyAttentionInputs::kv_cache_block_id)
        .def_readwrite("kv_cache_block_id_device", &PyAttentionInputs::kv_cache_block_id_device)
        .def_readwrite("dtype", &PyAttentionInputs::dtype)
        .def_readwrite("cu_seqlens_device", &PyAttentionInputs::cu_seqlens_device)
        .def_readwrite("cu_seqlens", &PyAttentionInputs::cu_seqlens)
        .def_readwrite("cu_kv_seqlens_device", &PyAttentionInputs::cu_kv_seqlens_device)
        .def_readwrite("context_total_kv_length", &PyAttentionInputs::context_total_kv_length)
        .def_readwrite("total_tokens", &PyAttentionInputs::total_tokens)
        .def_readwrite("padding_offset", &PyAttentionInputs::padding_offset)
        .def_readwrite("is_s_padded", &PyAttentionInputs::is_s_padded)
        .def_readonly("prefix_lengths_device", &PyAttentionInputs::prefix_lengths_device)
        .def_readwrite("sequence_lengths_plus_1_device", &PyAttentionInputs::sequence_lengths_plus_1_device)
        .def_readonly("input_lengths_device", &PyAttentionInputs::input_lengths_device)
        .def_readwrite("decode_cu_seqlens_device", &PyAttentionInputs::decode_cu_seqlens_device)
        .def_readwrite("decode_cu_seqlens", &PyAttentionInputs::decode_cu_seqlens)
        .def_readonly("cache_store_inputs", &PyAttentionInputs::cache_store_inputs)
        .def_readonly("cache_store_writer", &PyAttentionInputs::cache_store_writer)
        .def_readwrite("context_parallel_info", &PyAttentionInputs::context_parallel_info)
        .def_readwrite("combo_position_ids", &PyAttentionInputs::combo_position_ids)
        .def("__repr__", [](const PyAttentionInputs& self) { return "PyAttentionInputs"; })
        .def_readwrite("prefill_cuda_graph_copy_params", &PyAttentionInputs::prefill_cuda_graph_copy_params)
        .def_readwrite("headwise_config", &PyAttentionInputs::headwise_config)
        .def("__copy__", [](const PyAttentionInputs& self) { return PyAttentionInputs(self); });

    pybind11::class_<BertEmbeddingInputs>(m, "BertEmbeddingInputs")
        .def(pybind11::init<>())
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float>(),
             pybind11::arg("combo_position_ids")     = torch::empty(0),
             pybind11::arg("position_encoding")      = torch::empty(0),
             pybind11::arg("combo_tokens_type_ids")  = torch::empty(0),
             pybind11::arg("token_type_embedding")   = torch::empty(0),
             pybind11::arg("input_embedding_scalar") = 1.0f)
        .def_readwrite("combo_position_ids", &BertEmbeddingInputs::combo_position_ids, "Combined position IDs tensor")
        .def_readwrite("position_encoding", &BertEmbeddingInputs::position_encoding, "Position encoding tensor")
        .def_readwrite(
            "combo_tokens_type_ids", &BertEmbeddingInputs::combo_tokens_type_ids, "Combined token type IDs tensor")
        .def_readwrite(
            "token_type_embedding", &BertEmbeddingInputs::token_type_embedding, "Token type embedding tensor")
        .def_readwrite(
            "input_embedding_scalar", &BertEmbeddingInputs::input_embedding_scalar, "Input embedding scalar value")
        .def("__repr__", [](const BertEmbeddingInputs& self) { return "BertEmbeddingInputs"; });

    pybind11::class_<PyEmbeddingInputs>(m, "PyEmbeddingInputs")
        .def(pybind11::init<>())
        .def_readwrite(
            "combo_tokens_type_ids", &PyEmbeddingInputs::combo_tokens_type_ids, "Combined token type IDs tensor")
        .def_readwrite("text_tokens_mask", &PyEmbeddingInputs::text_tokens_mask, "Text tokens mask tensor")
        .def("__repr__", [](const PyEmbeddingInputs& self) { return "PyEmbeddingInputs"; });

    pybind11::class_<PyMultimodalInputs>(m, "PyMultimodalInputs")
        .def(pybind11::init<>())
        .def_readwrite("mm_features_spans", &PyMultimodalInputs::mm_features_spans)
        .def_readwrite("multimodal_features", &PyMultimodalInputs::multimodal_features, "Multimodal features tensor")
        .def_readwrite(
            "mm_features_locs", &PyMultimodalInputs::mm_features_locs, "Multimodal features locations tensor")
        .def_readwrite(
            "mm_extra_input", &PyMultimodalInputs::mm_extra_input, "Multimodal model-specific extra input tensor")
        .def("__repr__", [](const PyMultimodalInputs& self) { return "PyMultimodalInputs"; });

    pybind11::class_<PyModelInputs>(m, "PyModelInputs")
        .def(pybind11::init<>())
        .def(pybind11::init([](torch::Tensor       input_ids,
                               torch::Tensor       input_hiddens,
                               torch::Tensor       combo_position_ids,
                               PyEmbeddingInputs   embedding_inputs,
                               PyMultimodalInputs  multimodal_inputs,
                               pybind11::object    attention_inputs,
                               BertEmbeddingInputs bert_embedding_inputs) {
                 PyModelInputs result;
                 result.input_ids             = std::move(input_ids);
                 result.input_hiddens         = std::move(input_hiddens);
                 result.combo_position_ids    = std::move(combo_position_ids);
                 result.embedding_inputs      = std::move(embedding_inputs);
                 result.multimodal_inputs     = std::move(multimodal_inputs);
                 result.bert_embedding_inputs = std::move(bert_embedding_inputs);
                 if (pybind11::isinstance<PyAttentionInputs>(attention_inputs)) {
                     result.attention_inputs = attention_inputs.cast<PyAttentionInputs>();
                 } else {
                     result.attention_inputs_by_tag = attention_inputs.cast<AttentionInputsByTag>();
                     RTP_LLM_CHECK_WITH_INFO(!result.attention_inputs_by_tag.empty(),
                                             "attention_inputs tag map must not be empty");
                     result.attention_inputs = result.attention_inputs_by_tag.begin()->second;
                 }
                 return result;
             }),
             pybind11::arg("input_ids")             = torch::empty(0),
             pybind11::arg("input_hiddens")         = torch::empty(0),
             pybind11::arg("combo_position_ids")    = torch::empty(0),
             pybind11::arg("embedding_inputs")      = PyEmbeddingInputs(),
             pybind11::arg("multimodal_inputs")     = PyMultimodalInputs(),
             pybind11::arg("attention_inputs")      = PyAttentionInputs(),
             pybind11::arg("bert_embedding_inputs") = BertEmbeddingInputs())
        .def_readwrite("need_all_logits", &PyModelInputs::need_all_logits)
        .def_readwrite("need_all_hidden_states", &PyModelInputs::need_all_hidden_states)
        .def_readwrite("is_fake_stream", &PyModelInputs::is_fake_stream)
        .def_readwrite("warmup", &PyModelInputs::warmup)
        .def_readwrite("engram_token_windows", &PyModelInputs::engram_token_windows)
        .def_readwrite("prefill_replay_token_ids", &PyModelInputs::prefill_replay_token_ids)
        .def_readwrite("engram_token_windows_host",
                       &PyModelInputs::engram_token_windows_host,
                       "CP-local host history retained only for normal prefill")
        .def_readwrite("expert_token_counts", &PyModelInputs::expert_token_counts)
        .def_readwrite("input_ids", &PyModelInputs::input_ids, "Input token IDs tensor")
        .def_readwrite("request_id", &PyModelInputs::request_id, "Optional scheduler identities for speculative state")
        .def_readwrite("input_hiddens", &PyModelInputs::input_hiddens, "Input hidden states tensor")
        .def_readwrite("combo_position_ids", &PyModelInputs::combo_position_ids, "Combo position IDs tensor")
        .def_readwrite("embedding_inputs", &PyModelInputs::embedding_inputs, "Embedding inputs structure")
        .def_readwrite("multimodal_inputs", &PyModelInputs::multimodal_inputs, "Multimodal inputs structure")
        .def_property(
            "attention_inputs",
            [](PyModelInputs& self) -> pybind11::object {
                if (!self.attention_inputs_by_tag.empty()) {
                    pybind11::dict result;
                    for (auto& [tag, inputs] : self.attention_inputs_by_tag) {
                        result[pybind11::str(tag)] = pybind11::cast(
                            &inputs, pybind11::return_value_policy::reference_internal, pybind11::cast(&self));
                    }
                    return std::move(result);
                }
                return pybind11::cast(
                    &self.attention_inputs, pybind11::return_value_policy::reference_internal, pybind11::cast(&self));
            },
            [](PyModelInputs& self, pybind11::object value) {
                if (pybind11::isinstance<PyAttentionInputs>(value)) {
                    self.attention_inputs        = value.cast<PyAttentionInputs>();
                    self.attention_inputs_by_tag = {};
                    return;
                }
                auto by_tag = value.cast<AttentionInputsByTag>();
                RTP_LLM_CHECK_WITH_INFO(!by_tag.empty(), "attention_inputs tag map must not be empty");
                self.attention_inputs        = by_tag.begin()->second;
                self.attention_inputs_by_tag = std::move(by_tag);
            },
            "A PyAttentionInputs value or a tag-to-PyAttentionInputs mapping")
        .def_readwrite(
            "bert_embedding_inputs", &PyModelInputs::bert_embedding_inputs, "BERT embedding inputs structure")
        .def_readwrite("capture_hidden_states",
                       &PyModelInputs::capture_hidden_states,
                       "Capture configured hidden states for this homogeneous target-prefill batch");

    pybind11::class_<PyModelOutputs>(m, "PyModelOutputs")
        .def(pybind11::init<>(), "Default constructor")
        .def(pybind11::init<torch::Tensor>(), pybind11::arg("hidden_states"), "Initialize with hidden states tensor")
        .def(pybind11::init<torch::Tensor, torch::Tensor>(),
             pybind11::arg("hidden_states"),
             pybind11::arg("mtp_target_hidden_states"),
             "Initialize with hidden states and speculative target features")
        .def_readwrite("hidden_states", &PyModelOutputs::hidden_states, "Hidden states output tensor")
        .def_readwrite("logits", &PyModelOutputs::logits, "Optional local-vocabulary decode graph logits")
        .def(pybind11::init<torch::Tensor, torch::Tensor, torch::Tensor>(),
             pybind11::arg("hidden_states"),
             pybind11::arg("mtp_target_hidden_states"),
             pybind11::arg("dspark_confidence_hidden_logits"),
             "Initialize with hidden states, target features and scalar V4.1 confidence contributions")
        .def_readwrite("mtp_target_hidden_states",
                       &PyModelOutputs::mtp_target_hidden_states,
                       "Optional target features consumed by speculative decoding")
        .def_readwrite("dspark_confidence_hidden_logits",
                       &PyModelOutputs::dspark_confidence_hidden_logits,
                       "Optional FP32 [tokens, 1] pre-norm hidden contribution to V4.1 confidence logits");
}

}  // namespace torch_ext
