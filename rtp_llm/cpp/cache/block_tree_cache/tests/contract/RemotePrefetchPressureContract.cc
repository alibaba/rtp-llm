#include "block_tree_cache.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {
using block_tree_cache::Batch;
using block_tree_cache::CacheRef;
using block_tree_cache::Plan;
using block_tree_cache::bytes;

const char* case_name = "";

void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << case_name << ": " << message << ": " << btc_last_error() << '\n';
        std::abort();
    }
}

struct Fixture {
    explicit Fixture(size_t slots = 6, std::vector<btc_group> groups = {{0, BTC_FULL, 1, 0, 1, nullptr, 0, 0}},
                     bool release_local = true, bool writeback = true):
        memory(slots, 0) {
        const uint32_t rank = 0, storage = 4;
        std::vector<btc_route> routes{
            {BTC_WORK_DEVICE_TO_HOST, &rank, 1}, {BTC_WORK_HOST_TO_DEVICE, &rank, 1},
            {BTC_WORK_REMOTE_QUERY, &storage, 1}, {BTC_WORK_REMOTE_TO_HOST, &storage, 1}};
        if (writeback) {
            routes.push_back({BTC_WORK_HOST_TO_REMOTE, &storage, 1});
        }
        btc_config config{};
        config.abi_version = btc_abi_version();
        config.name_space = bytes("remote pressure");
        config.key_tokens = 1;
        config.groups = groups.data();
        config.group_count = groups.size();
        config.host_slots = slots;
        config.routes = routes.data();
        config.route_count = routes.size();
        config.release_local_pages = release_local;
        const btc_host_region region{{memory.data(), memory.size()}, 1, 1};
        check(btc_open_with_host_region(&config, &region, &handle) == BTC_OK, "open mapped Host cache");
    }

    void close() {
        while (btc_drain(handle) == BTC_PENDING) {
            const auto batch = poll();
            const auto view = btc_batch_get(batch.get());
            check(view.work_count > 0, "drain owns outstanding work");
            for (size_t i = 0; i < view.work_count; ++i) {
                check(view.work[i].kind == BTC_WORK_RELEASE, "all writes and reads have settled");
                finish(view.work[i]);
            }
        }
        check(btc_close(handle) == BTC_OK, "close");
    }

    Batch poll() const {
        Batch batch;
        check(CacheRef(handle).poll(batch) == BTC_OK, "poll");
        return batch;
    }

    Plan match(uint64_t request, std::initializer_list<std::string_view> keys, btc_status expected,
               bool remote = true, size_t max_loads = SIZE_MAX, const uint32_t* selected = nullptr) const {
        std::vector<btc_bytes> path;
        for (auto key : keys) {
            path.push_back(bytes(key));
        }
        const btc_limits limits{keys.size(), selected, selected ? 1u : 0u, nullptr, 0, max_loads, remote};
        Plan plan;
        check(CacheRef(handle).match(request, {path.data(), path.size()}, limits, plan) == expected, "match");
        check(static_cast<bool>(plan) == (expected == BTC_OK), "only a final match returns a plan");
        return plan;
    }

    Plan pollMatch(uint64_t request, btc_status expected = BTC_OK) const {
        Plan plan;
        check(CacheRef(handle).pollMatch(request, plan) == expected, "poll the admitted match");
        check(static_cast<bool>(plan) == (expected == BTC_OK), "only a final poll returns a plan");
        return plan;
    }

    void complete(btc_operation operation, uint32_t actor, btc_result result, bool safe, bool committed,
                  const std::vector<btc_remote_object>& objects = {}) const {
        const btc_receipt receipt{operation, actor, result, safe, committed, objects.data(), objects.size()};
        check(btc_complete(handle, &receipt, 1) == BTC_OK, "receipt");
    }

    void finish(const btc_work& command, const std::vector<btc_remote_object>& objects = {}) const {
        for (size_t i = 0; i < command.participant_count; ++i) {
            complete(command.id, command.participants[i], BTC_SUCCESS, true, true, objects);
        }
    }

    static const btc_work& only(const Batch& batch, btc_work_kind kind, size_t pages) {
        const auto view = btc_batch_get(batch.get());
        if (view.work_count != 1 || view.work[0].kind != kind || view.work[0].page_count != pages) {
            std::cerr << "Expected one work item with kind=" << kind << " pages=" << pages
                      << ". Actual items=" << view.work_count << '\n';
            for (size_t i = 0; i < view.work_count; ++i) {
                std::cerr << "  kind=" << view.work[i].kind << " pages=" << view.work[i].page_count << '\n';
            }
            check(false, "unexpected transfer batch");
        }
        return view.work[0];
    }

    static const btc_work& find(const Batch& batch, btc_work_kind kind) {
        const auto view = btc_batch_get(batch.get());
        for (size_t i = 0; i < view.work_count; ++i) {
            if (view.work[i].kind == kind) {
                return view.work[i];
            }
        }
        check(false, "expected a pending operation of this kind");
        std::abort();
    }

    void release(const Batch& batch, uint64_t pool, size_t pages) {
        const auto& command = only(batch, BTC_WORK_RELEASE, pages);
        for (size_t i = 0; i < pages; ++i) {
            check(command.pages[i].source_count == 1 && command.pages[i].source[0].pool == pool,
                  "release belongs to the expected pool");
        }
        finish(command);
    }

    bool event(const Batch& batch, uint64_t request, btc_event_kind kind) const {
        const auto view = btc_batch_get(batch.get());
        for (size_t i = 0; i < view.event_count; ++i) {
            if (view.events[i].request == request && view.events[i].kind == kind) {
                return true;
            }
        }
        return false;
    }

    uint8_t& host(btc_position position, btc_page page) {
        btc_buffer buffer{};
        check(page.pool == BTC_HOST_POOL
                  && btc_host_slot_buffer(handle, position.group, page.slot, &buffer) == BTC_OK
                  && buffer.size == 1, "resolve real Host bytes");
        return *static_cast<uint8_t*>(buffer.data);
    }

    void resident(std::string_view key, uint64_t index, uint32_t group = 0) {
        residentPath({key}, index, group);
    }

    void residentPath(const std::vector<std::string_view>& keys, uint64_t index,
                      uint32_t group = 0, bool concurrent = false) {
        const btc_page source{1, index, 1, index};
        const btc_binding row{{group, keys.size()}, &source, 1};
        std::vector<btc_bytes> path;
        for (const auto key : keys) {
            path.push_back(bytes(key));
        }
        check(btc_publish(handle, {path.data(), path.size()}, path.size(), &row, 1) == BTC_OK, "publish Device page");
        if (!concurrent) {
            check(btc_batch_get(poll().get()).work_count == 0, "publish alone leaves Device unmodified");
        }
        check(btc_demote(handle, source.pool, 1, &source, 1, 1) == BTC_OK, "demote eligible Device page");
        const auto staging = poll();
        const auto& copy = find(staging, BTC_WORK_DEVICE_TO_HOST);
        check(copy.page_count == 1, "demote only the admitted Device page");
        host(copy.pages[0].position, copy.pages[0].target[0]) = static_cast<uint8_t>(index);
        finish(copy);
        const auto settled = poll();
        const auto& returned = find(settled, BTC_WORK_RELEASE);
        check(returned.page_count == 1 && returned.pages[0].source[0].pool == source.pool,
              "demotion returns only the original Device page");
        finish(returned);
    }

    void fillHost() {
        for (uint64_t i = 1; i <= 5; ++i) {
            resident("old-" + std::to_string(i), i);
        }
    }

    void replyToQuery(size_t pages) {
        const auto batch = poll();
        const auto& query = only(batch, BTC_WORK_REMOTE_QUERY, pages);
        std::vector<btc_remote_object> objects;
        for (size_t i = 0; i < query.page_count; ++i) {
            objects.push_back({query.pages[i].position, bytes("committed remote replica")});
        }
        finish(query, objects);
    }

    std::vector<uint8_t> memory;
    btc_cache* handle = nullptr;
};

void four_pages_with_one_initial_slot() {
    Fixture cache;
    cache.fillHost();
    const std::initializer_list<std::string_view> keys{"k1", "k2", "k3", "k4"};
    cache.match(1, keys, BTC_PENDING);
    cache.match(2, keys, BTC_PENDING);
    cache.replyToQuery(4);
    auto batch = cache.poll();
    const auto& spill = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 3);
    std::set<uint64_t> spilled_slots;
    std::set<std::string> spilled_keys;
    for (size_t i = 0; i < spill.page_count; ++i) {
        check(cache.host(spill.pages[i].position, spill.pages[i].source[0]) == i + 1,
              "writeback preserves the selected cold Host bytes until safe completion");
        spilled_slots.insert(spill.pages[i].source[0].slot);
        check(spill.pages[i].position.group == 0 && spill.pages[i].position.end_token == 1,
              "the cold Host victims share a group and end position");
        const auto& key = spill.pages[i].key;
        spilled_keys.emplace(static_cast<const char*>(key.data), key.size);
    }
    check(spilled_slots.size() == 3, "only the foreground deficit is written back");
    check(spilled_keys == std::set<std::string>{"old-1", "old-2", "old-3"},
          "distinct keys at the same depth survive Host writeback batching");
    check(btc_cancel(cache.handle, 1) == BTC_OK, "cancelling one joined request preserves the other");
    cache.finish(spill);
    batch = cache.poll();
    cache.pollMatch(2, BTC_PENDING);
    check(cache.event(batch, 1, BTC_EVENT_CANCELLED), "cancel notification is local to the first waiter");
    cache.release(batch, BTC_HOST_POOL, 3);
    batch = cache.poll();
    const auto& read = Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 4);
    std::set<uint64_t> targets;
    for (size_t i = 0; i < read.page_count; ++i) {
        check(read.pages[i].target_count == 1, "one Host destination per remote page");
        const auto page = read.pages[i].target[0];
        cache.host(read.pages[i].position, page) = 20 + read.pages[i].position.end_token;
        targets.insert(page.slot);
    }
    check(targets.size() == 4, "all four destinations were reserved together");
    cache.finish(read);
    check(cache.event(cache.poll(), 2, BTC_EVENT_PROGRESS), "surviving waiter resumes after the batch warms");
    const auto plan = cache.pollMatch(2);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 0 && view.recoverable_tokens == 4, "confirmed Host prefix is ready to bind");
    std::array<btc_page, 4> device{};
    std::vector<btc_binding> bindings;
    for (size_t i = 0; i < device.size(); ++i) {
        device[i] = {1, 20 + i, 1, 20 + i};
        bindings.push_back({{0, i + 1}, &device[i], 1});
    }
    check(btc_bind(cache.handle, plan.get(), bindings.data(), bindings.size()) == BTC_WAITING, "bind four Device pages");
    batch = cache.poll();
    const auto& load = Fixture::only(batch, BTC_WORK_HOST_TO_DEVICE, 4);
    for (size_t i = 0; i < load.page_count; ++i) {
        check(cache.host(load.pages[i].position, load.pages[i].source[0]) == 20 + load.pages[i].position.end_token,
              "Host to Device uses the exact bytes returned by the remote read");
    }
    cache.finish(load);
    batch = cache.poll();
    check(cache.event(batch, 2, BTC_EVENT_READY), "all four Device pages settle together");
    cache.release(batch, BTC_HOST_POOL, 4);
    const auto local = cache.match(3, keys, BTC_OK, false);
    check(btc_plan_get(local.get()).ready_tokens == 4, "successful migration leaves only Device as the serving tier");
    cache.close();
}

void shared_pool_uses_peer_victims_and_keeps_the_active_host_prefix() {
    Fixture cache(6, {{0, BTC_FULL, 1, 0, 1, nullptr, 0, 0}, {1, BTC_FULL, 1, 0, 1, nullptr, 0, 0}}, false);
    cache.resident("prefix", 1);
    for (uint64_t i = 2; i <= 5; ++i) {
        cache.resident("peer-" + std::to_string(i), i, 1);
    }
    const uint32_t group = 0;
    const std::initializer_list<std::string_view> keys{"prefix", "k1", "k2", "k3", "k4"};
    cache.match(1, keys, BTC_PENDING, true, SIZE_MAX, &group);
    cache.replyToQuery(4);
    auto batch = cache.poll();
    const auto& spill = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 3);
    for (size_t i = 0; i < spill.page_count; ++i) {
        check(spill.pages[i].position.group == 1, "foreground cannot evict its active Host prefix");
    }
    cache.finish(spill);
    batch = cache.poll();
    cache.finish(Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 4));
    const auto plan = cache.pollMatch(1);
    check(btc_plan_get(plan.get()).recoverable_tokens == 5, "peer Host slots extend the protected local prefix");
    cache.close();
}

void admission_protects_device_and_host_until_bind_or_cancel() {
    Fixture cache(2);
    const btc_page device{1, 1, 1, 1};
    const btc_bytes prefix[]{bytes("device")};
    const btc_binding ready{{0, 1}, &device, 1};
    check(btc_publish(cache.handle, {prefix, 1}, 1, &ready, 1) == BTC_OK, "publish Device prefix");
    cache.residentPath({"device", "host"}, 2);

    const auto pressure = [&] {
        check(btc_reclaim(cache.handle, device.pool, 1, &device, 1) == BTC_OK, "apply Device allocation pressure");
        check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 1) == BTC_OK, "apply Host pressure");
    };
    std::string remote_key = "remote";
    const auto admit = [&](uint64_t request) {
        const std::array path{bytes("device"), bytes("host"), bytes(remote_key), bytes("beyond-token-budget")};
        const btc_limits limits{path.size(), nullptr, 0, nullptr, 0, SIZE_MAX, 1};
        btc_plan* plan = nullptr;
        check(btc_match_capped(cache.handle, request, {path.data(), path.size()}, &limits, 2, &plan) == BTC_PENDING && !plan,
              "admit a two-token suffix relative to the Device hit");
    };
    cache.pollMatch(10, BTC_RETRY);
    admit(10);
    admit(11);
    remote_key.front() = 'x';
    pressure();
    cache.pollMatch(11, BTC_PENDING);
    cache.replyToQuery(1);

    auto batch = cache.poll();
    const auto& read = Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 1);
    check(read.pages[0].position.end_token == 3, "read only the missing tail");
    cache.host(read.pages[0].position, read.pages[0].target[0]) = 3;
    cache.finish(read);

    check(btc_cancel(cache.handle, 10) == BTC_OK, "detach one waiter after the read settles");
    cache.pollMatch(10, BTC_RETRY);
    pressure();
    check(btc_batch_get(cache.poll().get()).work_count == 0,
          "the surviving admission protects Device and both Host rows before any final plan is polled");
    const auto plan = cache.pollMatch(11);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 1 && view.host_tokens == 3 && view.recoverable_tokens == 3 && view.page_count == 3
              && view.pages[0].state == BTC_PAGE_READY && view.pages[0].page_count == 1
              && view.pages[0].pages[0].hold == device.hold
              && view.pages[1].route == BTC_WORK_HOST_TO_DEVICE && view.pages[2].route == BTC_WORK_HOST_TO_DEVICE,
          "final plan retains the exact original Device and Host matches");
    cache.pollMatch(11, BTC_RETRY);
    pressure();
    check(btc_batch_get(cache.poll().get()).work_count == 0, "final plan keeps its pages until bind");

    // Simulate the caller failing to allocate either Device destination.
    check(btc_cancel(cache.handle, 11) == BTC_OK, "cancel an offered match when Device space cannot be reserved");
    cache.pollMatch(11, BTC_RETRY);
    const btc_page targets[]{{1, 3, 1, 3}, {1, 4, 1, 4}};
    const btc_binding late[]{{{0, 2}, targets, 1}, {{0, 3}, targets + 1, 1}};
    check(btc_bind(cache.handle, plan.get(), late, 2) == BTC_RETRY, "the canceled plan cannot bind");

    pressure();
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 2, "cancellation makes both tiers available to pressure");
    const auto& device_release = Fixture::find(batch, BTC_WORK_RELEASE);
    check(device_release.page_count == 1 && device_release.pages[0].source[0].hold == device.hold,
          "retire the original Device hold exactly once");
    cache.finish(device_release);
    const auto& tail = Fixture::find(batch, BTC_WORK_HOST_TO_REMOTE);
    check(tail.page_count == 1 && tail.pages[0].position.end_token == 3, "retire the warmed Host tail");
    cache.finish(tail);
    cache.release(cache.poll(), BTC_HOST_POOL, 1);

    check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 1) == BTC_OK, "reclaim the original Host prefix");
    batch = cache.poll();
    const auto& old = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 1);
    check(old.pages[0].position.end_token == 2 && cache.host(old.pages[0].position, old.pages[0].source[0]) == 2,
          "canceled admission no longer pins the original Host row");
    cache.finish(old);
    cache.release(cache.poll(), BTC_HOST_POOL, 1);

    const auto retry = cache.match(11, {"device", "host", "remote"}, BTC_OK, false);
    check(btc_plan_get(retry.get()).recoverable_tokens == 0, "reuse the request ID with a fresh admission");
    cache.close();
}

void completed_join_preserves_the_selected_device_target() {
    Fixture cache;
    cache.resident("head", 1);
    auto owner = cache.match(5, {"head"}, BTC_OK, false);
    const btc_page head{1, 8, 1, 8};
    const btc_binding load_head{{0, 1}, &head, 1};
    check(btc_bind(cache.handle, owner.get(), &load_head, 1) == BTC_WAITING, "start the original Host load");
    auto batch = cache.poll();
    const auto loading = Fixture::only(batch, BTC_WORK_HOST_TO_DEVICE, 1).id;
    check(btc_accept(cache.handle, loading, 0) == BTC_OK, "claim the first Host load before admitting its waiter");

    cache.match(6, {"head", "tail"}, BTC_PENDING);
    batch = cache.poll();
    const auto& query = Fixture::find(batch, BTC_WORK_REMOTE_QUERY);
    check(query.page_count == 1 && query.pages[0].position.end_token == 2, "query only the remote tail");
    cache.finish(query, {{{0, 2}, bytes("remote tail")}});
    batch = cache.poll();
    const auto& warming = Fixture::find(batch, BTC_WORK_REMOTE_TO_HOST);
    check(warming.page_count == 1 && warming.pages[0].position.end_token == 2, "read only the remote tail");
    cache.complete(loading, 0, BTC_SUCCESS, true, false);
    cache.finish(warming);

    const auto waiter = cache.pollMatch(6);
    const auto view = btc_plan_get(waiter.get());
    check(view.recoverable_tokens == 2 && view.page_count == 2 && view.pages[0].state == BTC_PAGE_READY
              && view.pages[0].page_count == 1 && view.pages[0].pages[0].hold == head.hold
              && view.pages[1].state == BTC_PAGE_LOAD && view.pages[1].route == BTC_WORK_HOST_TO_DEVICE,
          "the waiter retains the exact completed JOIN and still needs one allocator target");
    const btc_page tail{1, 9, 1, 9};
    const btc_binding load_tail{{0, 2}, &tail, 1};
    check(btc_bind(cache.handle, waiter.get(), &load_tail, 1) == BTC_WAITING, "bind only the tail");
    batch = cache.poll();
    const auto& load = Fixture::find(batch, BTC_WORK_HOST_TO_DEVICE);
    check(load.page_count == 1 && load.pages[0].position.end_token == 2, "the completed JOIN is never copied again");
    cache.finish(load);
    cache.close();
}

void admitted_host_source_survives_inflight_writeback() {
    Fixture cache(2);
    cache.resident("head", 1);
    check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 1) == BTC_OK, "start a Host writeback before admission");
    auto batch = cache.poll();
    const auto& writeback = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 1);
    const btc_page source = writeback.pages[0].source[0];
    const auto in_flight = writeback.id;
    cache.match(7, {"head", "tail"}, BTC_PENDING);
    batch = cache.poll();
    const auto& query = Fixture::find(batch, BTC_WORK_REMOTE_QUERY);
    check(query.page_count == 1 && query.pages[0].position.end_token == 2, "retain the in-flight Host prefix");
    cache.finish(query, {{{0, 2}, bytes("remote tail")}});

    cache.complete(in_flight, 4, BTC_SUCCESS, true, true);
    batch = cache.poll();
    cache.finish(Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 1));
    const auto plan = cache.pollMatch(7);
    const auto view = btc_plan_get(plan.get());
    check(view.recoverable_tokens == 2 && view.page_count == 2 && view.pages[0].state == BTC_PAGE_LOAD
              && view.pages[0].page_count == 1
              && view.pages[0].pages[0].slot == source.slot && view.pages[0].pages[0].generation == source.generation
              && cache.host(view.pages[0].position, view.pages[0].pages[0]) == 1,
          "a later writeback receipt cannot remove or replace an admitted Host source");
    cache.close();
}

void chained_full_victims_continue_after_competing_admission() {
    Fixture cache(6, {{0, BTC_FULL, 1, 0, 1, nullptr, 0, 0}}, false);
    std::vector<std::string_view> path;
    for (const auto key : {"chain-1", "chain-2", "chain-3", "chain-4", "chain-5"}) {
        path.push_back(key);
        cache.residentPath(path, path.size());
    }
    const std::initializer_list<std::string_view> keys{"k1", "k2", "k3", "k4"};
    cache.match(1, keys, BTC_PENDING);
    cache.replyToQuery(4);
    Plan competitor;
    for (size_t depth = 5; depth >= 2; --depth) {
        const auto batch = cache.poll();
        const auto& spill = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 1);
        check(spill.pages[0].position.end_token == depth, "FULL eviction exposes one new leaf per committed write");
        cache.finish(spill);
        if (depth == 5) {
            // A normal demotion uses one free slot while the next leaf write
            // is in flight. Its pending load keeps that slot active, so the
            // cold read must obtain one more slot from the FULL chain.
            cache.residentPath({"competitor"}, 6, 0, true);
            competitor = cache.match(99, {"competitor"}, BTC_OK, false);
            check(btc_plan_get(competitor.get()).recoverable_tokens == 1, "competitor owns an active Host load");
        }
    }
    const auto batch = cache.poll();
    cache.finish(Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 4));
    check(btc_plan_get(cache.pollMatch(1).get()).recoverable_tokens == 4,
          "foreground waits for the complete batch despite a competing allocation");
    cache.close();
}

void writeback_failure_closes_wait_without_reusing_unsafe_slots() {
    for (const bool safe : {false, true}) {
        Fixture cache;
        cache.fillHost();
        const std::initializer_list<std::string_view> keys{"k1", "k2", "k3", "k4"};
        cache.match(1, keys, BTC_PENDING);
        cache.replyToQuery(4);
        auto batch = cache.poll();
        const auto& spill = Fixture::only(batch, BTC_WORK_HOST_TO_REMOTE, 3);
        const auto operation = spill.id;
        cache.complete(operation, 4, safe ? BTC_FAILED : BTC_TIMEOUT, safe, false);
        batch = cache.poll();
        check(cache.event(batch, 1, BTC_EVENT_PROGRESS), "foreground waiter finishes on failed or uncertain write");
        const auto view = btc_batch_get(batch.get());
        check(view.work_count == (safe ? 0u : 1u), "only the uncertain write remains owned");
        if (!safe) {
            check(view.work[0].id.operation == operation.operation
                      && btc_drain(cache.handle) == BTC_PENDING,
                  "unknown ownership keeps the original operation and blocks close");
            cache.complete(operation, 4, BTC_FAILED, true, false);
        } else {
            const auto fallback = cache.pollMatch(1);
            check(btc_plan_get(fallback.get()).recoverable_tokens == 0, "failed writeback falls back without a second query");
            const auto original = cache.match(2, {"old-1"}, BTC_OK, false);
            check(btc_plan_get(original.get()).recoverable_tokens == 1, "failed writeback preserves original Host data");
        }
        cache.close();
    }
}

void cancelled_read_cannot_notify_a_reused_request_id() {
    Fixture cache;
    const btc_page resident{1, 9, 1, 9};
    const btc_bytes stable[]{bytes("ready")};
    const btc_binding ready{{0, 1}, &resident, 1};
    check(btc_publish(cache.handle, {stable, 1}, 1, &ready, 1) == BTC_OK, "publish an unrelated READY page");
    const auto old_ready = cache.match(7, {"ready"}, BTC_OK, false);
    check(btc_plan_get(old_ready.get()).ready_tokens == 1, "the caller still owns a delivered READY plan");
    cache.match(7, {"old"}, BTC_PENDING);
    check(btc_bind(cache.handle, old_ready.get(), nullptr, 0) == BTC_RETRY,
          "a previously delivered plan cannot bind over a new pending admission with the same ID");
    cache.pollMatch(7, BTC_PENDING);
    cache.replyToQuery(1);
    auto batch = cache.poll();
    const auto old = Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 1).id;
    check(btc_cancel(cache.handle, 7) == BTC_OK, "cancel the first remote Host read");
    cache.match(7, {"new"}, BTC_PENDING);
    batch = cache.poll();
    const auto query = Fixture::find(batch, BTC_WORK_REMOTE_QUERY).id;
    const auto delivered = btc_batch_get(batch.get());
    check(cache.event(batch, 7, BTC_EVENT_CANCELLED) && delivered.event_count > 0,
          "the first attempt reported cancellation");
    check(btc_ack_events(cache.handle, delivered.events[delivered.event_count - 1].cursor) == BTC_OK,
          "acknowledge previously delivered events before checking the replacement");
    cache.complete(old, 4, BTC_TIMEOUT, false, false);
    cache.pollMatch(7, BTC_PENDING);
    batch = cache.poll();
    check(!cache.event(batch, 7, BTC_EVENT_PROGRESS), "late old timeout cannot finish the new waiter");
    cache.complete(old, 4, BTC_FAILED, true, false);
    batch = cache.poll();
    const auto& release = Fixture::find(batch, BTC_WORK_RELEASE);
    check(release.page_count == 1 && release.pages[0].source[0].pool == BTC_HOST_POOL,
          "late safe receipt retires only the old Host destination");
    cache.finish(release);
    cache.complete(query, 4, BTC_SUCCESS, true, false, {{{0, 1}, bytes("new replica")}});
    batch = cache.poll();
    cache.finish(Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 1));
    check(btc_plan_get(cache.pollMatch(7).get()).recoverable_tokens == 1,
          "new request with the same id completes independently");
    cache.close();
}

void elapsed_deadline_releases_the_waiter_before_any_write_receipt() {
    Fixture cache;
    cache.fillHost();
    const std::initializer_list<std::string_view> keys{"k1", "k2", "k3", "k4"};
    cache.match(1, keys, BTC_PENDING);
    cache.replyToQuery(4);
    const auto scheduled = cache.poll();
    const auto write = Fixture::only(scheduled, BTC_WORK_HOST_TO_REMOTE, 3).id;
    // The public C ABI uses the native default 10-second Host deadline.
    std::this_thread::sleep_for(std::chrono::milliseconds(10100));
    const auto expired = cache.poll();
    check(cache.event(expired, 1, BTC_EVENT_PROGRESS)
              && Fixture::only(expired, BTC_WORK_HOST_TO_REMOTE, 3).id.operation == write.operation,
          "elapsed time releases the foreground waiter while the writer still owns its source");
    check(btc_plan_get(cache.pollMatch(1).get()).recoverable_tokens == 0,
          "a timed-out cold match falls back without another remote query");
    check(btc_drain(cache.handle) == BTC_PENDING, "the actual writer still prevents shutdown");
    cache.complete(write, 4, BTC_FAILED, true, false);
    cache.close();
}

void shorten_to_the_largest_legal_prefix() {
    Fixture cache(5, {{0, BTC_FULL, 1, 0, 1, nullptr, 0, 0},
                      {1, BTC_LINEAR, 2, 0, 1, nullptr, 0, 0},
                      {2, BTC_SWA, 1, 2, 1, nullptr, 0, 0}}, false);
    const std::initializer_list<std::string_view> keys{"k1", "k2", "k3", "k4"};
    cache.match(1, keys, BTC_PENDING);
    cache.replyToQuery(10);
    const auto batch = cache.poll();
    const auto& read = Fixture::only(batch, BTC_WORK_REMOTE_TO_HOST, 5);
    for (size_t i = 0; i < read.page_count; ++i) {
        check(read.pages[i].position.end_token <= 2, "no page beyond the shortened legal endpoint is read");
    }
    cache.finish(read);
    const auto plan = cache.pollMatch(1);
    check(btc_plan_get(plan.get()).recoverable_tokens == 2, "FULL, LINEAR and SWA agree on the largest fitting boundary");
    cache.close();
}

void capped_swa_reports_the_selected_host_endpoint() {
    Fixture cache(4, {{0, BTC_SWA, 1, 2, 1, nullptr, 0, 0}}, false);
    const std::array keys{bytes("k1"), bytes("k2"), bytes("k3"), bytes("k4")};
    std::vector<std::string_view> path;
    for (const auto key : {"k1", "k2", "k3", "k4"}) {
        path.push_back(key);
        cache.residentPath(path, path.size());
    }
    const btc_limits limits{keys.size(), nullptr, 0, nullptr, 0, SIZE_MAX, 0};
    btc_plan* raw = nullptr;
    check(btc_match_capped(cache.handle, 1, {keys.data(), keys.size()}, &limits, 2, &raw) == BTC_OK && raw,
          "select the capped Host window");
    const Plan plan(raw);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 0 && view.host_tokens == 2 && view.recoverable_tokens == 2 && view.page_count == 2,
          "report Host coverage for the capped window instead of the original four-token window");
    for (size_t i = 0; i < view.page_count; ++i) {
        const auto& page = view.pages[i];
        check(page.route == BTC_WORK_HOST_TO_DEVICE && page.position.end_token == i + 1
                  && page.page_count == 1 && cache.host(page.position, page.pages[0]) == i + 1,
              "the capped window uses the original first two Host pages");
    }
    cache.close();
}

}  // namespace

int main() {
    const auto run = [](const char* name, void (*test)()) {
        case_name = name;
        test();
    };
    run("four pages with one initial slot", four_pages_with_one_initial_slot);
    run("shared pool preserves active Host prefix", shared_pool_uses_peer_victims_and_keeps_the_active_host_prefix);
    run("admission protects Device and Host until bind or cancel", admission_protects_device_and_host_until_bind_or_cancel);
    run("completed JOIN preserves the chosen Device target", completed_join_preserves_the_selected_device_target);
    run("admitted Host source survives in-flight writeback", admitted_host_source_survives_inflight_writeback);
    run("FULL eviction chain under competing admission", chained_full_victims_continue_after_competing_admission);
    run("unsafe and safe writeback failures", writeback_failure_closes_wait_without_reusing_unsafe_slots);
    run("late receipt after request ID reuse", cancelled_read_cannot_notify_a_reused_request_id);
    run("elapsed foreground deadline", elapsed_deadline_releases_the_waiter_before_any_write_receipt);
    run("longest legal FULL/LINEAR/SWA prefix", shorten_to_the_largest_legal_prefix);
    run("capped SWA preserves Host attribution", capped_swa_reports_the_selected_host_endpoint);
}
