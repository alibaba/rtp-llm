#include "pybind11/pybind11.h"

#if USING_CUDA
#include "rtp_llm/cpp/pybind/emb_local/EmbHashCache.h"
#include "rtp_llm/cpp/pybind/emb_local/EmbLocalShard.h"

#include <pybind11/stl.h>
#include <unordered_map>
#include <unordered_set>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>

namespace rtp_llm {
namespace {
namespace py = pybind11;
using namespace emb_local;

template<typename Function>
auto withoutGil(Function function) {
    py::gil_scoped_release release;
    return function();
}

Shard::Streams streamFor(uintptr_t stream, int device) {
    if (device < 0) {
        return {};
    }
    return c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream), device);
}

py::tuple lookupResult(const Index::Lookup& state) {
    const auto* tier = state.resident ? (state.resident->tier == Tier::Gpu ? "gpu" : "cpu") : "";
    return py::make_tuple(state.cell, tier, state.cell ? state.cell->generation() : 0);
}

bool encode(py::handle                              value,
            Tree&                                   tree,
            Shard::TensorList&                      tensors,
            std::unordered_map<PyObject*, int64_t>& seen,
            size_t                                  depth) {
    if (depth > 64) {
        return false;
    }
    if (value.is_none()) {
        return true;
    }
    using Kind = Tree::Kind;
    auto* obj  = value.ptr();
    if (THPVariable_Check(obj)) {
        tree.kind                    = Kind::Tensor;
        const auto [entry, inserted] = seen.try_emplace(obj, tensors.size());
        tree.integer                 = entry->second;
        if (inserted) {
            tensors.push_back(py::cast<at::Tensor>(value));
        }
        return true;
    }
    if (PyTuple_CheckExact(obj) || PyList_CheckExact(obj)) {
        const bool tuple = PyTuple_CheckExact(obj);
        tree.kind        = tuple ? Kind::Tuple : Kind::List;
        const auto count = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
        tree.items.resize(count);
        for (Py_ssize_t i = 0; i < count; ++i) {
            auto* item = tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
            if (!encode(item, tree.items[i], tensors, seen, depth + 1)) {
                return false;
            }
        }
        return true;
    }
    if (PyDict_CheckExact(obj)) {
        tree.kind = Kind::Dict;
        tree.items.reserve(PyDict_GET_SIZE(obj) * 2);
        PyObject*  key;
        PyObject*  item;
        Py_ssize_t offset = 0;
        while (PyDict_Next(obj, &offset, &key, &item)) {
            tree.items.resize(tree.items.size() + 2);
            if (!encode(key, tree.items[tree.items.size() - 2], tensors, seen, depth + 1)
                || !encode(item, tree.items.back(), tensors, seen, depth + 1)) {
                return false;
            }
        }
        return true;
    }
    if (PyBool_Check(obj)) {
        tree.kind    = Kind::Bool;
        tree.integer = obj == Py_True;
    } else if (PyLong_CheckExact(obj)) {
        int overflow = 0;
        tree.kind    = Kind::Integer;
        tree.integer = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (overflow) {
            return false;
        }
    } else if (PyFloat_CheckExact(obj)) {
        tree.kind = Kind::Float;
        tree.real = PyFloat_AS_DOUBLE(obj);
    } else if (PyUnicode_CheckExact(obj)) {
        tree.kind = Kind::String;
        tree.text = py::cast<std::string>(value);
    } else if (PyBytes_CheckExact(obj)) {
        tree.kind = Kind::Bytes;
        tree.text.assign(PyBytes_AS_STRING(obj), PyBytes_GET_SIZE(obj));
    } else {
        return false;
    }
    return true;
}

py::object decode(const Tree& tree, const py::tuple& tensors) {
    using Kind = Tree::Kind;
    switch (tree.kind) {
        case Kind::None:
            return py::none();
        case Kind::Tensor:
            return tensors[tree.integer];
        case Kind::Bool:
            return py::bool_(tree.integer != 0);
        case Kind::Integer:
            return py::int_(tree.integer);
        case Kind::Float:
            return py::float_(tree.real);
        case Kind::String:
            return py::str(tree.text);
        case Kind::Bytes:
            return py::bytes(tree.text);
        case Kind::Tuple: {
            py::tuple result(tree.items.size());
            for (size_t i = 0; i < tree.items.size(); ++i) {
                result[i] = decode(tree.items[i], tensors);
            }
            return result;
        }
        case Kind::List: {
            py::list result(tree.items.size());
            for (size_t i = 0; i < tree.items.size(); ++i) {
                result[i] = decode(tree.items[i], tensors);
            }
            return result;
        }
        case Kind::Dict: {
            py::dict result;
            for (size_t i = 0; i < tree.items.size(); i += 2) {
                auto key    = decode(tree.items[i], tensors);
                result[key] = decode(tree.items[i + 1], tensors);
            }
            return result;
        }
    }
    throw std::logic_error("invalid local EMB value tree");
}

void setPoolStats(py::dict& output, const char* prefix, const Arena::Stats& stats) {
    const auto put = [&](const char* field, int64_t value) { output[py::str(std::string(prefix) + field)] = value; };
    put("capacity_bytes", stats.capacity);
    put("free_bytes", stats.free);
    put("used_bytes", stats.capacity - stats.free);
    put("reserved_bytes", stats.reserved);
    put("resident_bytes", stats.resident);
    put("pending_bytes", stats.retired);
    put("quarantined_bytes", stats.quarantined);
    put("wasted_bytes", 0);
    put("largest_free_block_bytes", stats.largest_free);
}

template<typename... Args>
py::tuple mutationResult(Shard::Mutations& mutations, Args&&... args) {
    if (mutations.snapshots.empty()) {
        return py::make_tuple(std::forward<Args>(args)...);
    }
    return py::make_tuple(std::forward<Args>(args)..., std::move(mutations.snapshots));
}
}  // namespace

void registerEmbLocalCache(pybind11::module& m) {
    py::class_<RemoteWriteQueue, std::shared_ptr<RemoteWriteQueue>>(m, "EmbRemoteWriteQueue")
        .def(py::init([](std::shared_ptr<EmbRegisteredArena> arena, std::optional<at::Tensor> root, py::capsule capsule,
                        int64_t object_bytes, int64_t max_pending, int64_t max_bytes, int64_t timeout_ms,
                        int workers, bool request_scoped) {
            const auto* api = static_cast<const KvcmObjectNativeApiV2*>(
                PyCapsule_GetPointer(capsule.ptr(), KVCM_OBJECT_NATIVE_API_CAPSULE_V2));
            if (!api) {
                throw py::error_already_set();
            }
            if (!validEmbNativeWriteApi(api)) {
                throw std::invalid_argument("invalid exact-object native API");
            }
            return std::make_shared<RemoteWriteQueue>(std::move(arena), root.value_or(at::Tensor{}), *api, object_bytes, max_pending,
                                                     max_bytes, timeout_ms, workers, request_scoped);
        }), py::keep_alive<1, 4>())
        .def("reserve_external", &RemoteWriteQueue::reserveExternal, py::call_guard<py::gil_scoped_release>())
        .def("release_external", &RemoteWriteQueue::releaseExternal, py::call_guard<py::gil_scoped_release>())
        .def("close", &RemoteWriteQueue::close, py::call_guard<py::gil_scoped_release>())
        .def("submit_encoded", [](RemoteWriteQueue& writer,
                                  const std::string& key,
                                  const std::string& trace,
                                  py::handle value,
                                  const std::optional<std::tuple<bool, bool, int64_t, std::string>>& verdict,
                                  const std::vector<std::pair<c10::DeviceIndex, c10::StreamId>>& producers) {
            Tree tree;
            Shard::TensorList tensors;
            std::unordered_map<PyObject*, int64_t> seen;
            if (!encode(value, tree, tensors, seen, 0)) {
                return;
            }
            Audit audit;
            if (verdict) {
                audit.verdict_present = true;
                std::tie(audit.checked, audit.passed, audit.code, audit.message) = *verdict;
            }
            py::gil_scoped_release release;
            writer.submitEncoded(key, trace, audit, tree, tensors, producers);
        })
        .def("stats", [](const RemoteWriteQueue& writer) {
            const auto stats = withoutGil([&] { return writer.stats(); });
            return py::make_tuple(stats.pending, stats.bytes, stats.success, stats.error, stats.skipped,
                                  stats.inflight_failure);
        });
    m.def("lock_emb_write_capture", [](bool wait) {
        auto& gate = RemoteWriteQueue::graphGate();
        if (!wait) {
            return gate.try_lock();
        }
        withoutGil([&] { gate.lock(); });
        return true;
    });
    m.def("unlock_emb_write_capture", [] { RemoteWriteQueue::graphGate().unlock(); });
    py::class_<Shard::Eviction>(m, "EmbLocalEviction")
        .def("unpack", [](const Shard::Eviction& victim) {
            py::tuple leaves(victim.tensors.size());
            for (size_t i = 0; i < victim.tensors.size(); ++i) {
                leaves[i] = py::cast(victim.tensors[i]);
            }
            const auto& audit = victim.audit;
            return py::make_tuple(victim.key, victim.remote_key, victim.generation,
                                  victim.tier == Tier::Cpu ? "cpu" : "gpu", decode(*victim.tree, leaves),
                                  audit.origin, audit.verdict_present, audit.checked, audit.passed, audit.code,
                                  audit.message);
        });
    py::class_<Cell, std::shared_ptr<Cell>>(m, "EmbLocalGeneration")
        .def_property_readonly("generation", &Cell::generation)
        .def("set_verdict", &Cell::setVerdict)
        .def("set_origin", &Cell::setOrigin)
        .def("set_remote_key", &Cell::setRemoteKey)
        .def("set_audit_error", &Cell::setAuditError)
        .def("audit", [](const Cell& cell) {
            const auto audit = cell.audit();
            return py::make_tuple(audit.decided,
                                  audit.checked,
                                  audit.verdict_present,
                                  audit.passed,
                                  audit.code,
                                  audit.message,
                                  audit.origin);
        });
    py::class_<Ready, std::shared_ptr<Ready>>(m, "EmbLocalProducerReady")
        .def("wait", &Ready::wait, py::call_guard<py::gil_scoped_release>());
    m.def("record_emb_local_ready", [](const std::vector<std::pair<uintptr_t, int>>& streams) {
        std::vector<std::shared_ptr<Ready>> events;
        events.reserve(streams.size());
        for (const auto& [stream, device] : streams) {
            events.push_back(std::make_shared<Ready>(streamFor(stream, device).value()));
        }
        return events;
    });
    py::class_<Shard>(m, "EmbLocalCacheShard")
        .def(py::init<int64_t, int64_t, int>())
        .def(py::init<int64_t, int64_t, int, bool, bool, int64_t>())
        .def(py::init<int64_t, int64_t, int, bool, bool, int64_t, std::shared_ptr<RemoteWriteQueue>>(),
             py::keep_alive<1, 8>())
        .def("claim",
             [](Shard& shard, const std::string& key) {
                 return lookupResult(withoutGil([&] { return shard.claim(key); }));
             })
        .def("peek",
             [](Shard& shard, const std::string& key) {
                 return lookupResult(withoutGil([&] { return shard.lookup(key, false); }));
             })
        .def("is_current",
             [](Shard& shard, const std::string& key, const Cell& cell) {
                 return shard.withCurrent(key, cell, [] {});
             },
             py::call_guard<py::gil_scoped_release>())
        .def("with_current",
             [](Shard& shard, const std::string& key, const Cell& cell, py::function action) {
                 py::object result = py::none();
                 {
                     // The callback commits only generation-bound hash metadata.
                     // Index waiters must release the GIL so this callback can
                     // finish even if Python switches threads in the commit.
                     py::gil_scoped_release release;
                     shard.withCurrent(key, cell, [&] {
                         py::gil_scoped_acquire acquire;
                         result = action();
                     });
                 }
                 return result;
             })
        .def("erase",
             [](Shard& shard, const std::string& key, const Cell& cell) { return shard.erase(key, cell.generation()); },
             py::call_guard<py::gil_scoped_release>())
        .def("clear", &Shard::clear, py::call_guard<py::gil_scoped_release>())
        .def("close", &Shard::close, py::call_guard<py::gil_scoped_release>())
        .def("publish",
             [](Shard&                                     shard,
                const std::string&                         key,
                const Cell&                                cell,
                py::handle                                 tree_value,
                int64_t                                    tokens,
                const std::vector<std::shared_ptr<Ready>>& ready,
                uintptr_t                                  stream,
                int                                        device) -> py::tuple {
                 auto                                   tree = std::make_shared<Tree>();
                 Shard::TensorList                      tensors;
                 std::unordered_map<PyObject*, int64_t> seen;
                 bool                                   published = false;
                 Shard::Mutations                       mutations;
                 if (!encode(tree_value, *tree, tensors, seen, 0)) {
                     return py::make_tuple(false, int64_t{0}, int64_t{0}, int64_t{0});
                 }
                 {
                     py::gil_scoped_release release;
                     auto                   result = shard.publish(
                         key, cell.generation(), tensors, std::move(tree), tokens, ready, streamFor(stream, device));
                     published = result.first;
                     mutations = std::move(result.second);
                 }
                 auto       record = published ? cell.record() : nullptr;
                 const auto bytes  = record ? record->bytes[0] + record->bytes[1] : 0;
                 return mutationResult(mutations, published, bytes, mutations.evicted, mutations.demoted);
             })
        .def("read",
             [](Shard& shard, const std::string& key, const Cell& cell, uintptr_t stream, int device) -> py::tuple {
                 auto record = cell.record();
                 if (!record) {
                     return py::make_tuple(py::object(py::none()), false, int64_t{0}, int64_t{0}, false);
                 }
                 Shard::TensorList result;
                 bool              promoted         = false;
                 bool              promotion_failed = false;
                 Shard::Mutations  mutations;
                 {
                     py::gil_scoped_release release;
                     auto                   cuda_stream = streamFor(stream, device);
                     result                             = shard.read(record, cuda_stream);
                     if (record->tier == Tier::Cpu) {
                         try {
                             auto promotion = shard.tryPromote(key, record, result, cuda_stream);
                             promoted       = promotion.first;
                             mutations      = std::move(promotion.second);
                         } catch (const std::exception&) {
                             // Readback is already complete and independently owned.
                             promotion_failed = true;
                         }
                     }
                 }
                 py::tuple leaves(result.size());
                 for (size_t i = 0; i < result.size(); ++i) {
                     leaves[i] = py::cast(std::move(result[i]));
                 }
                 return mutationResult(mutations, decode(*record->tree, leaves), promoted, mutations.evicted,
                                       mutations.demoted, promotion_failed);
             })
        .def(
            "resident_tier",
            [](Shard& shard, const std::string& key, bool touch) -> const char* {
                const auto record = shard.lookup(key, touch).resident;
                if (!record) {
                    return nullptr;
                }
                return record->tier == Tier::Gpu ? "gpu" : "cpu";
            },
            py::call_guard<py::gil_scoped_release>())
        .def("resident_tiers",
             [](Shard& shard, const std::optional<std::vector<std::string>>& keys, int64_t limit) {
                 py::dict result;
                 if (limit == 0) {
                     return result;
                 }
                 const auto records = withoutGil([&] {
                     if (!keys) {
                         return shard.residency(limit);
                     }
                     std::vector<std::pair<std::string, Tier>> records;
                     std::unordered_set<std::string_view> seen;
                     for (const auto& key : *keys) {
                         if (!seen.insert(key).second) {
                             continue;
                         }
                         const auto record = shard.lookup(key, false).resident;
                         if (record) {
                             records.emplace_back(key, record->tier);
                             if (limit >= 0 && records.size() >= static_cast<size_t>(limit)) {
                                 break;
                             }
                         }
                     }
                     return records;
                 });
                 for (const auto& [key, tier] : records) {
                     result[py::str(key)] = tier == Tier::Gpu ? "gpu" : "cpu";
                 }
                 return result;
             })
        .def("resize",
             [](Shard& shard, int64_t gpu, int64_t cpu, uintptr_t stream, int device) {
                 Shard::Mutations result;
                 {
                     py::gil_scoped_release release;
                     result = shard.resize(gpu, cpu, streamFor(stream, device));
                 }
                 return mutationResult(result, result.evicted, result.demoted, result.cleared_through);
             })
        .def("resident_charge",
             [](Shard& shard) {
                 const auto snapshot = withoutGil([&] { return shard.charge(); });
                 return py::make_tuple(snapshot.tokens,
                                       snapshot.bytes[0] + snapshot.bytes[1],
                                       snapshot.entries[0] + snapshot.entries[1]);
             })
        .def("stats", [](Shard& shard) {
            const auto [snapshot, cpu, gpu] = withoutGil([&] {
                return std::make_tuple(shard.charge(), shard.stats(Tier::Cpu), shard.stats(Tier::Gpu));
            });
            py::dict out;
            out["resident_tokens"]      = snapshot.tokens;
            out["resident_bytes"]       = snapshot.bytes[0] + snapshot.bytes[1];
            out["resident_entries"]     = snapshot.entries[0] + snapshot.entries[1];
            out["gpu_resident_entries"] = snapshot.entries[1];
            out["cpu_resident_entries"] = snapshot.entries[0];
            out["gpu_resident_bytes"]   = snapshot.bytes[1];
            out["cpu_resident_bytes"]   = snapshot.bytes[0];
            out["pending_entries"]      = snapshot.pending;
            out["eviction_snapshot_dropped"] = shard.droppedSnapshots();
            setPoolStats(out, "cpu_pool_", cpu);
            setPoolStats(out, "gpu_pool_", gpu);
            return out;
        });
    const auto release = py::call_guard<py::gil_scoped_release>();
    py::class_<HashCache>(m, "EmbLocalHashCache")
        .def(py::init<int64_t>())
        .def("put", &HashCache::put, release)
        .def("put_for_entry",
             [](HashCache& hashes,
                Shard& shard,
                const Cell& cell,
                const std::string& key,
                const HashCache::Hashes& tensors,
                const std::string& generation,
                bool approved) {
                 hashes.putForGeneration(key, tensors, generation, approved, true, [&](auto action) {
                     shard.withCurrent(key, cell, std::move(action));
                 });
             },
             release)
        .def("snapshot", &HashCache::snapshot, release)
        .def("metadata", &HashCache::metadata, release)
        .def("contains", &HashCache::contains, release)
        .def("approved", &HashCache::approved, release)
        .def("mark_approved", &HashCache::markApproved, release)
        .def("keys", &HashCache::keys, release)
        .def("clear", &HashCache::clear, release)
        .def("resize", &HashCache::resize, release)
        .def("enabled", &HashCache::enabled, release)
        .def("stats", &HashCache::stats, release);
}
}  // namespace rtp_llm
#endif
