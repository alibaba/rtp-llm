#include "block_tree_cache.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {
using block_tree_cache::Batch;
using block_tree_cache::CacheRef;
using block_tree_cache::Plan;
using block_tree_cache::bytes;

void check(bool passed, const char* message) {
    if (!passed) {
        std::cerr << message << " " << btc_last_error() << '\n';
        std::abort();
    }
}

struct Keys {
    explicit Keys(std::initializer_list<std::string_view> keys) {
        for (auto key : keys) {
            values.push_back(bytes(key));
        }
    }
    btc_keys get() const { return {values.data(), values.size()}; }
    std::vector<btc_bytes> values;
};

struct Cache {
    Cache(std::vector<btc_group> groups, uint64_t host = 0, uint64_t disk = 0,
          std::vector<btc_route> routes = {}, std::string_view directory = {},
          bool release_local = false, bool external_disk = false,
          const btc_host_region* host_region = nullptr) {
        const btc_config config{btc_abi_version(), bytes(std::string_view("contract\0v1", 11)), 8,
                                groups.data(), groups.size(), host, disk, bytes(directory),
                                routes.data(), routes.size(), release_local, BTC_EVICT_LRU, external_disk, 0};
        const auto status = host_region ? btc_open_with_host_region(&config, host_region, &handle)
                                        : btc_open(&config, &handle);
        check(status == BTC_OK, "open");
    }

    void publish(const Keys& keys, std::vector<btc_binding> bindings) const {
        check(btc_publish(handle, keys.get(), keys.values.size() * 8, bindings.data(), bindings.size()) == BTC_OK,
              "publish");
    }

    void demote(const btc_page& page, bool remove_device = true) const {
        check(btc_demote(handle, page.pool, 1, &page, 1, remove_device) == BTC_OK, "demote Device page");
    }

    Plan match(uint64_t request, const Keys& keys, uint64_t tokens, btc_status status = BTC_OK, bool remote = false) const {
        const btc_limits limits{tokens, nullptr, 0, nullptr, 0, SIZE_MAX, remote};
        Plan plan;
        check(CacheRef(handle).match(request, keys.get(), limits, plan) == status, "match");
        return plan;
    }

    Plan pollMatch(uint64_t request, btc_status expected = BTC_OK) const {
        Plan plan;
        check(CacheRef(handle).pollMatch(request, plan) == expected, "poll the admitted match");
        check(static_cast<bool>(plan) == (expected == BTC_OK), "only a final poll returns a plan");
        return plan;
    }

    Batch poll() const {
        Batch batch;
        check(CacheRef(handle).poll(batch) == BTC_OK, "poll");
        return batch;
    }

    void receipt(btc_operation id, uint32_t participant, btc_result result = BTC_SUCCESS,
                 bool safe = true, bool committed = false,
                 const btc_remote_object* objects = nullptr, size_t count = 0) const {
        const btc_receipt receipt{id, participant, result, safe, committed, objects, count};
        check(btc_complete(handle, &receipt, 1) == BTC_OK, "settle");
    }

    void finish(const btc_work& work) const {
        for (size_t i = 0; i < work.participant_count; ++i) {
            receipt(work.id, work.participants[i], BTC_SUCCESS, true, true);
        }
    }

    void invalidate(const btc_page& page) const {
        check(btc_invalidate(handle, &page, 1) == BTC_OK, "invalidate");
    }

    void finish_releases() const {
        auto batch = poll();
        const auto view = btc_batch_get(batch.get());
        for (size_t i = 0; i < view.work_count; ++i) {
            if (view.work[i].kind == BTC_WORK_RELEASE) {
                finish(view.work[i]);
            }
        }
    }

    void close() const {
        while (btc_drain(handle) == BTC_PENDING) {
            auto batch = poll();
            const auto view = btc_batch_get(batch.get());
            check(view.work_count != 0, "drain needs an unresolved operation");
            for (size_t i = 0; i < view.work_count; ++i) {
                check(view.work[i].kind == BTC_WORK_RELEASE, "test left unfinished work");
                finish(view.work[i]);
            }
        }
        check(btc_close(handle) == BTC_OK, "close");
    }

    btc_cache* handle = nullptr;
};

const btc_work& work(const Batch& batch, btc_work_kind kind, size_t page_count) {
    const auto view = btc_batch_get(batch.get());
    for (size_t i = 0; i < view.work_count; ++i) {
        if (view.work[i].kind == kind) {
            check(view.work[i].page_count == page_count, "work page count");
            return view.work[i];
        }
    }
    check(false, "expected work kind");
    std::abort();
}

bool has_event(const Batch& batch, uint64_t request, btc_event_kind kind) {
    const auto view = btc_batch_get(batch.get());
    for (size_t i = 0; i < view.event_count; ++i) {
        if (view.events[i].request == request && view.events[i].kind == kind) {
            return true;
        }
    }
    return false;
}

void geometry_and_holds() {
    const uint64_t members[]{13, 14};
    const uint64_t shared[]{13};
    Cache cache({{0, BTC_FULL, 8, 0, 2, members, 0, 0},
                 {1, BTC_LINEAR, 8, 0, 1, shared, 0, 0},
                 {2, BTC_SWA, 8, 16, 1, shared, 0, 0}});
    const Keys keys{"a", std::string_view("b\0suffix", 8), "c", "d"};
    const std::array<std::array<btc_page, 2>, 4> full{{
        {{{13, 14, 1, 1}, {14, 14, 1, 2}}},
        {{{13, 15, 1, 3}, {14, 15, 1, 4}}},
        {{{13, 16, 1, 5}, {14, 16, 1, 6}}},
        {{{13, 17, 1, 7}, {14, 17, 1, 8}}},
    }};
    cache.publish(keys, {{{0, 8}, full[0].data(), 2}, {{0, 16}, full[1].data(), 2},
                         {{0, 24}, full[2].data(), 2}, {{0, 32}, full[3].data(), 2},
                         {{1, 32}, full[3].data(), 1}, {{2, 24}, full[2].data(), 1},
                         {{2, 32}, full[3].data(), 1}});
    auto plan = cache.match(1, keys, 32);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 32 && view.recoverable_tokens == 32 && view.page_count == 7,
          "FULL, late LINEAR and SWA after a gap must reach the same boundary");
    check(btc_bind(cache.handle, plan.get(), nullptr, 0) == BTC_OK, "ready bind");
    check(btc_bind(cache.handle, plan.get(), nullptr, 0) == BTC_RETRY, "plan reuse");
    check(btc_reclaim(cache.handle, 13, 1, full[0].data(), 1) == BTC_OK, "reclaim internal full prefix");
    check(btc_batch_get(cache.poll().get()).work_count == 0, "FULL reclaim preserves parent");
    auto stale = cache.match(2, keys, 32);
    cache.invalidate(full[3][1]);
    check(btc_bind(cache.handle, stale.get(), nullptr, 0) == BTC_RETRY, "old ready plan was invalidated");
    auto release_batch = cache.poll();
    const auto& release = work(release_batch, BTC_WORK_RELEASE, 1);
    check(release.pages[0].source[0].pool == 14 && release.pages[0].source[0].hold == 8,
          "distinct pools both have slot 17; aliases retain physical hold 7");
    cache.finish(release);
    const Keys another{"unrelated", std::string_view("b\0suffix", 8), "c", "d"};
    auto unrelated = cache.match(3, another, 32);
    check(btc_plan_get(unrelated.get()).ready_tokens == 0, "hash edges keep distinct prefix identities");
    cache.close();
}

void reentrant_catalog_with_aliased_pages() {
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}});
    const Keys keys{std::string_view("alias\0first", 11), "second"};
    const btc_page shared{3, 4, 7, 22};
    check(btc_track_device_changes(cache.handle, 1) == BTC_OK, "track native device changes");
    cache.publish(keys, {{{0, 8}, &shared, 1}, {{0, 16}, &shared, 1}});
    auto plan = cache.match(1, keys, 16);
    check(btc_plan_get(plan.get()).ready_tokens == 16, "same physical hold at two logical rows");
    check(btc_bind(cache.handle, plan.get(), nullptr, 0) == BTC_OK, "bind aliased READY pages");
    const auto catalog = [](void* ctx, uint64_t version, const btc_catalog_item* items, size_t count) {
        const auto handle = static_cast<btc_cache*>(ctx);
        size_t keys = 0, groups = 0;
        check(btc_device_stats(handle, &keys, &groups) == BTC_OK && keys == 2 && groups == 2,
              "catalog callback can reenter C API and sees native counts");
        check(version > 0 && count == 2 && items[0].group == 0 && items[1].group == 0,
              "one catalog entry per logical row");
        const auto first = [](btc_bytes key) {
            return std::string_view(static_cast<const char*>(key.data), key.size) == std::string_view("alias\0first", 11);
        };
        check(first(items[0].key) || first(items[1].key), "catalog returns complete binary keys");
    };
    check(btc_read_device_catalog(cache.handle, catalog, cache.handle) == BTC_OK, "read native catalog");
    cache.invalidate(shared);
    auto batch = cache.poll();
    const auto& release = work(batch, BTC_WORK_RELEASE, 1);
    check(release.pages[0].source[0].hold == shared.hold, "release one physical hold for both logical rows");
    cache.finish(release);
    const auto changes = [](void*, const btc_device_change* items, size_t count) {
        check(count == 4 && items[0].added && items[1].added && !items[2].added && !items[3].added,
              "native tree tracks real device additions and removals");
    };
    check(btc_read_device_changes(cache.handle, changes, nullptr) == BTC_OK, "read native device changes");
    cache.close();
}

void reclaim_uses_the_requested_physical_pool() {
    const uint64_t pools[]{10, 20};
    Cache cache({{7, BTC_FULL, 8, 0, 2, pools, 0, 0}});
    const std::array<std::array<btc_page, 2>, 3> members{{
        {{{10, 9, 1, 1}, {20, 9, 1, 2}}},
        {{{10, 10, 1, 3}, {20, 10, 1, 4}}},
        {{{10, 11, 1, 5}, {20, 11, 1, 6}}},
    }};
    const char* keys[]{"a", "b", "c"};
    btc_page eligible[3];
    for (size_t i = 0; i < members.size(); ++i) {
        cache.publish(Keys{keys[i]}, {{{7, 8}, members[i].data(), 2}});
        eligible[i] = members[i][1];
    }
    check(btc_reclaim(cache.handle, 20, 1, eligible, 3) == BTC_OK, "reclaim one slot from the second Device member");
    auto batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 2));
    check(btc_reclaim(cache.handle, 20, 2, eligible, 3) == BTC_OK,
          "pressure is counted in one pool and ignores the retired slot");
    batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 4));
    for (size_t i = 0; i < members.size(); ++i) {
        check(btc_plan_get(cache.match(i + 1, Keys{keys[i]}, 8).get()).ready_tokens == 0,
              "each grouped value was retired as a unit");
    }
    cache.close();
}

void reclaim_keeps_all_aliases_until_unpinned() {
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}});
    const btc_page first{20, 9, 1, 1};
    const btc_page second{20, 9, 1, 2};
    cache.publish(Keys{"a"}, {{{0, 8}, &first, 1}});
    cache.publish(Keys{"b"}, {{{0, 8}, &first, 1}});
    cache.publish(Keys{"c"}, {{{0, 8}, &second, 1}});
    auto pending = cache.match(1, Keys{"b"}, 8);
    check(btc_reclaim(cache.handle, 20, 1, &first, 1) == BTC_OK, "one aliased resource has a request reference");
    check(btc_batch_get(cache.poll().get()).work_count == 0, "physical aliases cannot be partially reclaimed");
    pending.reset();
    check(btc_reclaim(cache.handle, 20, 1, &first, 1) == BTC_OK, "retire all aliases of the physical slot");
    const auto batch = cache.poll();
    const auto& release = work(batch, BTC_WORK_RELEASE, 2);
    check(release.pages[0].source[0].hold != release.pages[1].source[0].hold, "each distinct physical hold releases once");
    cache.finish(release);
    cache.close();
}

void reclaim_counts_all_returned_slots_in_one_pool() {
    const uint64_t members[]{20, 20};
    Cache cache({{0, BTC_FULL, 8, 0, 2, members, 0, 0}});
    const btc_page pages[]{{20, 9, 1, 1}, {20, 10, 1, 2}, {20, 11, 1, 3}, {20, 12, 1, 4}};
    cache.publish(Keys{"a"}, {{{0, 8}, pages, 2}});
    cache.publish(Keys{"b"}, {{{0, 8}, pages + 2, 2}});
    const btc_page eligible[]{pages[0], pages[2]};
    check(btc_reclaim(cache.handle, 20, 2, eligible, 2) == BTC_OK,
          "an atomic drop returns both members even when only one can select the group");
    const auto batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 2));
    check(btc_plan_get(cache.match(1, Keys{"b"}, 8).get()).ready_tokens == 8, "stop once the pool recovers two slots");
    cache.close();
}

void related_reclaim_honors_cross_pool_selection() {
    const uint64_t members[]{10, 20};
    Cache cache({{0, BTC_FULL, 8, 0, 2, members, 0, 0}, {1, BTC_LINEAR, 8, 0, 1, nullptr, 0, 0}});
    const Keys keys{"related"};
    const btc_page pages[]{{10, 5, 1, 1}, {20, 5, 1, 2}, {30, 5, 1, 3}};
    cache.publish(keys, {{{0, 8}, pages, 2}, {{1, 8}, &pages[2], 1}});
    check(btc_reclaim_related(cache.handle, 20, 1, &pages[1], 1) == BTC_OK, "partial root selection");
    check(btc_batch_get(cache.poll().get()).work_count == 0, "unlisted companion prevents related reclaim");
    const uint32_t sibling_group = 1;
    const btc_limits limits{8, &sibling_group, 1, nullptr, 0, SIZE_MAX, 0};
    Plan sibling;
    check(CacheRef(cache.handle).match(1, keys.get(), limits, sibling) == BTC_OK, "pin sibling group");
    check(btc_reclaim_related(cache.handle, 20, 1, pages, 3) == BTC_OK, "accept eligible pages from three pools");
    auto batch = cache.poll();
    const auto& release = work(batch, BTC_WORK_RELEASE, 2);
    check(release.pages[0].source[0].pool != 30 && release.pages[1].source[0].pool != 30,
          "a pinned sibling remains cached");
    cache.finish(release);
    check(btc_bind(cache.handle, sibling.get(), nullptr, 0) == BTC_OK, "sibling plan remains valid");

    const btc_page newer[]{{10, 5, 2, 4}, {20, 5, 2, 5}, pages[2]};
    cache.publish(keys, {{{0, 8}, newer, 2}});
    check(btc_reclaim_related(cache.handle, 20, 1, newer, 3) == BTC_OK, "eligible sibling can be removed");
    batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 3));
    cache.close();
}

void publication_and_load_move_only_the_selected_group() {
    const uint32_t workers[]{1, 4};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}, {1, BTC_LINEAR, 8, 0, 1, nullptr, 0, 0}}, 1, 0,
                {{BTC_WORK_DEVICE_TO_HOST, workers, 2}, {BTC_WORK_HOST_TO_DEVICE, workers, 2}}, {}, true);
    const Keys keys{"overlapping-groups"};
    const btc_page first{20, 3, 1, 1};
    const btc_page peer{21, 3, 1, 2};
    const std::vector<btc_binding> published{{{0, 8}, &first, 1}, {{1, 8}, &peer, 1}};
    cache.publish(keys, published);
    cache.publish(keys, published);
    check(btc_batch_get(cache.poll().get()).work_count == 0, "publishing the same Device pages queues no copies");
    {
        const auto plan = cache.match(1, keys, 8);
        const auto view = btc_plan_get(plan.get());
        check(view.ready_tokens == 8 && view.ready_page_count == 2, "both overlapping logical groups are ready");
    }

    cache.demote(first);
    auto batch = cache.poll();
    const auto& move = work(batch, BTC_WORK_DEVICE_TO_HOST, 1);
    const auto host = move.pages[0].target[0];
    check(move.pages[0].position.group == 0, "only the selected physical Device pool demotes");
    cache.finish(move);
    batch = cache.poll();
    const auto& device_release = work(batch, BTC_WORK_RELEASE, 1);
    check(device_release.pages[0].source[0].hold == first.hold, "demotion releases just its Device source");
    cache.finish(device_release);

    const auto plan = cache.match(2, keys, 8);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 0 && view.host_tokens == 8 && view.recoverable_tokens == 8 && view.page_count == 2,
          "the same node is recoverable across different group tiers");
    check(view.pages[0].position.group == 0 && view.pages[0].state == BTC_PAGE_LOAD
              && view.pages[0].route == BTC_WORK_HOST_TO_DEVICE && view.pages[1].position.group == 1
              && view.pages[1].state == BTC_PAGE_READY && view.pages[1].pages[0].hold == peer.hold,
          "only the demoted group needs a target allocation");
    const btc_page failed_target{20, 3, 2, 3};
    const btc_binding failing_binding{{0, 8}, &failed_target, 1};
    check(btc_bind(cache.handle, plan.get(), &failing_binding, 1) == BTC_WAITING, "bind only the Host group");
    batch = cache.poll();
    const auto failed = work(batch, BTC_WORK_HOST_TO_DEVICE, 1).id;
    cache.receipt(failed, 1);
    cache.receipt(failed, 4, BTC_FAILED);
    batch = cache.poll();
    const auto& target_release = work(batch, BTC_WORK_RELEASE, 1);
    check(target_release.pages[0].source[0].hold == failed_target.hold
              && has_event(batch, 2, BTC_EVENT_RECOMPUTE), "safe failure retires only the unsuccessful target");
    cache.finish(target_release);

    const auto retry = cache.match(3, keys, 8);
    check(btc_plan_get(retry.get()).recoverable_tokens == 8 && btc_plan_get(retry.get()).ready_tokens == 0,
          "the original Host source is still recoverable after a failed worker copy");
    const btc_page restored{20, 3, 3, 4};
    const btc_binding binding{{0, 8}, &restored, 1};
    check(btc_bind(cache.handle, retry.get(), &binding, 1) == BTC_WAITING, "retry the Host group");
    batch = cache.poll();
    const auto& load = work(batch, BTC_WORK_HOST_TO_DEVICE, 1);
    check(load.pages[0].source[0].slot == host.slot && load.pages[0].source[0].generation == host.generation,
          "retry loads the same Host allocation");
    const auto id = load.id;
    cache.receipt(id, 1);
    cache.receipt(id, 4, BTC_SUCCESS, false);
    batch = cache.poll();
    const auto awaiting = btc_batch_get(batch.get());
    check(awaiting.work_count == 1 && awaiting.work[0].kind == BTC_WORK_HOST_TO_DEVICE
              && !has_event(batch, 3, BTC_EVENT_READY), "the Host source remains leased while any worker is unsafe");
    cache.receipt(id, 4);
    batch = cache.poll();
    const auto& host_release = work(batch, BTC_WORK_RELEASE, 1);
    const auto released = host_release.pages[0].source[0];
    check(released.pool == BTC_HOST_POOL && released.slot == host.slot && released.generation == host.generation
              && has_event(batch, 3, BTC_EVENT_READY), "Device installation and Host release settle together");
    cache.finish(host_release);

    cache.publish(keys, {{{0, 8}, &restored, 1}, {{1, 8}, &peer, 1}});
    check(btc_batch_get(cache.poll().get()).work_count == 0, "repeat publication does not recreate a Host copy");
    const auto ready = cache.match(4, keys, 8);
    check(btc_plan_get(ready.get()).ready_tokens == 8, "both groups remain ready on Device");
    cache.close();
}

void shared_load_unsafe_failure() {
    const uint32_t ranks[]{1, 4};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 2, 0,
                {{BTC_WORK_HOST_TO_DEVICE, ranks, 2}, {BTC_WORK_DEVICE_TO_HOST, ranks, 2}}, {}, true);
    const Keys keys{"shared"};
    const btc_page source{10, 17, 10, 10};
    cache.publish(keys, {{{0, 8}, &source, 1}});
    cache.demote(source);
    auto batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_DEVICE_TO_HOST, 1));
    batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 1));

    auto plan = cache.match(1, keys, 8);
    const auto view = btc_plan_get(plan.get());
    check(view.recoverable_tokens == 8 && view.ready_tokens == 0 && view.pages[0].state == BTC_PAGE_LOAD,
          "Host hit");
    const btc_page target{10, 17, 11, 11};
    const btc_binding binding{{0, 8}, &target, 1};
    check(btc_bind(cache.handle, plan.get(), &binding, 1) == BTC_WAITING, "bind load");
    auto join = cache.match(2, keys, 8);
    check(btc_plan_get(join.get()).pages[0].state == BTC_PAGE_JOIN, "shared load");
    check(btc_bind(cache.handle, join.get(), nullptr, 0) == BTC_WAITING, "join shared operation");
    check(btc_cancel(cache.handle, 1) == BTC_OK, "cancel original waiter");
    batch = cache.poll();
    const auto id = work(batch, BTC_WORK_HOST_TO_DEVICE, 1).id;
    const btc_receipt contradictory[]{
        {id, 1, BTC_FAILED, 0, 0, nullptr, 0},
        {id, 1, BTC_REJECTED, 1, 0, nullptr, 0},
    };
    check(btc_complete(cache.handle, contradictory, 2) == BTC_INVALID,
          "an accepted operation cannot prove buffer safety by claiming rejection in the same batch");
    check(!has_event(cache.poll(), 2, BTC_EVENT_RECOMPUTE), "invalid receipt batch leaves the waiter live");
    cache.receipt({id.instance + 1, id.operation}, 4, BTC_FAILED);
    cache.receipt(id, 1);
    cache.receipt(id, 1);
    cache.receipt(id, 4, BTC_SUCCESS, false);
    batch = cache.poll();
    check(!has_event(batch, 2, BTC_EVENT_READY)
              && btc_batch_get(batch.get()).work_count == 1
              && btc_batch_get(batch.get()).work[0].kind == BTC_WORK_HOST_TO_DEVICE,
          "successful business receipts do not install Device data without every worker's proof");
    cache.receipt(id, 4, BTC_TIMEOUT, false);
    batch = cache.poll();
    check(has_event(batch, 1, BTC_EVENT_CANCELLED) && has_event(batch, 2, BTC_EVENT_RECOMPUTE)
              && !has_event(batch, 2, BTC_EVENT_READY),
          "cancel is per request and last-rank failure is visible");
    check(btc_batch_get(batch.get()).work_count == 1, "unsafe target must keep its hold");
    check(btc_drain(cache.handle) == BTC_PENDING, "unsafe rank blocks shutdown");
    cache.receipt(id, 4, BTC_TIMEOUT, true);
    batch = cache.poll();
    const auto& release = work(batch, BTC_WORK_RELEASE, 1);
    check(btc_batch_get(batch.get()).work_count == 1 && release.pages[0].source[0].hold == target.hold,
          "late safety releases the failed target and retains the original Host allocation");
    cache.finish(release);
    cache.close();
}

void invalidate_inflight_target() {
    const uint32_t worker[]{0};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 0,
                {{BTC_WORK_HOST_TO_DEVICE, worker, 1}, {BTC_WORK_DEVICE_TO_HOST, worker, 1}});
    const Keys keys{"invalidate-target"};
    const btc_page source{10, 17, 1, 31};
    cache.publish(keys, {{{0, 8}, &source, 1}});
    cache.demote(source);
    auto batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_DEVICE_TO_HOST, 1));
    cache.finish_releases();

    const btc_page target{10, 18, 1, 32};
    const btc_binding binding{{0, 8}, &target, 1};
    auto plan = cache.match(1, keys, 8);
    check(btc_bind(cache.handle, plan.get(), &binding, 1) == BTC_WAITING, "dispatch Host load");
    batch = cache.poll();
    const auto& load = work(batch, BTC_WORK_HOST_TO_DEVICE, 1);
    cache.invalidate(target);
    check(btc_batch_get(cache.poll().get()).work_count == 1, "invalidation does not release a worker's target");
    cache.finish(load);
    batch = cache.poll();
    check(has_event(batch, 1, BTC_EVENT_RECOMPUTE) && !has_event(batch, 1, BTC_EVENT_READY),
          "a revoked target stays unpublished even after successful I/O");
    const auto& release = work(batch, BTC_WORK_RELEASE, 1);
    check(release.pages[0].source[0].hold == target.hold, "release retired target after safety");
    cache.finish(release);
    const auto reusable = cache.match(2, keys, 8);
    const auto retained = btc_plan_get(reusable.get());
    check(retained.ready_tokens == 0 && retained.recoverable_tokens == 8 && retained.page_count == 1
              && retained.pages[0].route == BTC_WORK_HOST_TO_DEVICE,
          "revoked Device target leaves the original Host source reusable");
    cache.close();
}

void remote_warm_and_shared_timeout() {
    const uint32_t owner[]{0};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 0,
                {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_REMOTE_QUERY, owner, 1},
                 {BTC_WORK_REMOTE_TO_HOST, owner, 1}});
    const Keys keys{std::string_view("remote\0suffix", 13)};
    cache.match(1, keys, 8, BTC_PENDING, true);
    cache.match(2, keys, 8, BTC_PENDING, true);
    auto batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1, "concurrent requests share remote queries");
    const auto& query = work(batch, BTC_WORK_REMOTE_QUERY, 1);
    check(query.pages[0].key.size == keys.values[0].size, "complete binary prefix in remote query");
    const std::string_view descriptor("uri\0crc", 7);
    const btc_remote_object object{{0, 8}, bytes(descriptor)};
    cache.receipt(query.id, 0, BTC_SUCCESS, true, false, &object, 1);
    cache.pollMatch(1, BTC_PENDING);
    cache.match(2, keys, 8, BTC_PENDING, true);  // Existing C callers can advance an identical admission.
    batch = cache.poll();
    const auto& warm = work(batch, BTC_WORK_REMOTE_TO_HOST, 1);
    check(warm.pages[0].remote.size == descriptor.size(), "opaque remote descriptor is preserved");
    const auto operation = warm.id;
    cache.receipt(operation, 0, BTC_TIMEOUT, false);
    batch = cache.poll();
    check(has_event(batch, 1, BTC_EVENT_PROGRESS) && has_event(batch, 2, BTC_EVENT_PROGRESS),
          "unsafe remote warm failure resumes both request matches");
    const auto plan = cache.pollMatch(1);
    check(btc_plan_get(plan.get()).recoverable_tokens == 0, "failed warm is invisible before safety");
    check(btc_drain(cache.handle) == BTC_PENDING, "failed remote warm retains Host lease");
    cache.receipt(operation, 0, BTC_TIMEOUT, true);
    cache.close();
}

void host_spill_failure_can_retry_after_access() {
    const uint32_t owner[]{0};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 0,
                {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_DEVICE_TO_HOST, owner, 1},
                 {BTC_WORK_HOST_TO_REMOTE, owner, 1}});
    const Keys first{"first"};
    const Keys second{"second"};
    const btc_page original{1, 6, 1, 1};
    const btc_page newer{1, 7, 1, 2};
    cache.publish(first, {{{0, 8}, &original, 1}});
    cache.demote(original);
    auto batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_DEVICE_TO_HOST, 1));
    cache.finish_releases();
    cache.publish(second, {{{0, 8}, &newer, 1}});
    cache.demote(newer);
    batch = cache.poll();
    const auto spill = work(batch, BTC_WORK_HOST_TO_REMOTE, 1).id;
    cache.receipt(spill, 0, BTC_TIMEOUT, false);
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1, "unsafe remote spill isolates its slot");
    cache.receipt(spill, 0, BTC_FAILED, true);
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 0, "remote spill failure does not spin");
    {
        const auto plan = cache.match(1, first, 8);
        const auto view = btc_plan_get(plan.get());
        check(view.ready_tokens == 0 && view.recoverable_tokens == 8 && view.pages[0].route == BTC_WORK_HOST_TO_DEVICE,
              "failed remote write retains original Host data");
    }
    cache.demote(newer);
    batch = cache.poll();
    const auto& retried = work(batch, BTC_WORK_HOST_TO_REMOTE, 1);
    check(retried.id.operation != spill.operation, "new pressure retries after Host access");
    cache.finish(retried);
    cache.demote(newer);
    batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_DEVICE_TO_HOST, 1));
    cache.close();
}

void partitioned_demotion_and_disk_round_trip() {
    const uint32_t owner[]{0};
    const std::string directory = std::filesystem::temp_directory_path().string();
    Cache cache({{1, BTC_FULL, 8, 0, 1, nullptr, 2, 0}, {0, BTC_FULL, 8, 0, 1, nullptr, 1, 1}},
                0, 0, {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_DEVICE_TO_HOST, owner, 1},
                       {BTC_WORK_HOST_TO_DISK, owner, 1}, {BTC_WORK_DISK_TO_HOST, owner, 1}},
                directory, true);
    const Keys first{"disk-origin"};
    const Keys second{"host-pressure"};
    const btc_page original{1, 6, 1, 1};
    const btc_page companion{2, 6, 1, 2};
    const btc_page newer{1, 7, 1, 3};
    cache.publish(first, {{{0, 8}, &original, 1}, {{1, 8}, &companion, 1}});
    check(btc_batch_get(cache.poll().get()).work_count == 0, "publication leaves local pages on Device");
    check(btc_demote(cache.handle, 1, 1, &original, 1, 1) == BTC_OK, "demote first group");
    check(btc_demote(cache.handle, 2, 1, &companion, 1, 0) == BTC_OK, "store second group");
    auto batch = cache.poll();
    const auto view = btc_batch_get(batch.get());
    check(view.work_count == 2 && view.work[0].pages[0].target[0].slot == 0
              && view.work[1].pages[0].target[0].slot == 1,
          "group Host slot ranges are unique and ordered by group ID");
    const auto first_host = view.work[0].pages[0].target[0];
    cache.finish(view.work[0]);
    cache.finish(view.work[1]);
    cache.finish_releases();
    cache.publish(second, {{{0, 8}, &newer, 1}});
    check(btc_demote(cache.handle, 1, 1, &newer, 1, 1) == BTC_OK, "Host pressure spills the old page");
    batch = cache.poll();
    const auto& disk = work(batch, BTC_WORK_HOST_TO_DISK, 1);
    const auto disk_page = disk.pages[0].target[0];
    check(disk_page.pool == BTC_DISK_POOL && first_host.pool == BTC_HOST_POOL && disk_page.slot == first_host.slot,
          "Host and Disk own separate pools even when their public slot numbers match");
    std::array<uint8_t, 4> source{5, 0, 4, 8};
    btc_buffer buffer{source.data(), source.size()};
    check(btc_execute_disk(cache.handle, disk.id, &buffer, 1) == BTC_OK, "write physical Disk page");
    cache.finish(disk);
    check(btc_demote(cache.handle, 1, 1, &newer, 1, 1) == BTC_OK, "demote after safe Disk spill");
    batch = cache.poll();
    const auto releases = btc_batch_get(batch.get());
    check(releases.work_count == 1 && releases.work[0].kind == BTC_WORK_RELEASE,
          "new Host generation waits until release acknowledgment");
    const auto& first_release = work(batch, BTC_WORK_RELEASE, 1);
    const auto old_release = first_release.id;
    check(first_release.pages[0].source[0].generation == first_host.generation
              && first_release.pages[0].source[0].pool == BTC_HOST_POOL,
          "only the old Host generation is retiring");
    cache.receipt(old_release, 0, BTC_FAILED, false);
    check(btc_demote(cache.handle, 1, 1, &newer, 1, 1) == BTC_OK, "retry demotion while release is unsafe");
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1
              && work(batch, BTC_WORK_RELEASE, 1).id.operation == old_release.operation,
          "unsafe release keeps the same operation and the slot occupied");
    cache.receipt(old_release, 0, BTC_FAILED, true);
    batch = cache.poll();
    const auto& release_retry = work(batch, BTC_WORK_RELEASE, 1);
    const auto retry = release_retry.id;
    check(retry.operation != old_release.operation
              && release_retry.pages[0].source[0].generation == first_host.generation,
          "safe failure retries release of the same Host generation");
    cache.receipt(old_release, 0);
    check(btc_demote(cache.handle, 1, 1, &newer, 1, 1) == BTC_OK, "retry demotion after a stale success");
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1 && work(batch, BTC_WORK_RELEASE, 1).id.operation == retry.operation,
          "stale success cannot acknowledge the retry");
    cache.receipt(retry, 0);
    check(btc_demote(cache.handle, 1, 1, &newer, 1, 1) == BTC_OK, "demote after Host release");
    batch = cache.poll();
    const auto& next_store = work(batch, BTC_WORK_DEVICE_TO_HOST, 1);
    const auto second_host = next_store.pages[0].target[0];
    check(second_host.slot == first_host.slot && second_host.generation != first_host.generation,
          "successful acknowledgment reuses the Host slot with a new generation");
    cache.finish(next_store);
    cache.finish_releases();
    cache.match(1, first, 8, BTC_PENDING);
    batch = cache.poll();
    const auto reclaiming = btc_batch_get(batch.get());
    check(reclaiming.work_count == 1 && reclaiming.work[0].kind == BTC_WORK_RELEASE,
          "Disk restore waits for the released Host slot");
    const auto& second_release = work(batch, BTC_WORK_RELEASE, 1);
    const auto next_release = second_release.id;
    check(second_release.pages[0].source[0].generation == second_host.generation,
          "later pressure retires only the replacement Host generation");
    cache.receipt(retry, 0);
    cache.pollMatch(1, BTC_PENDING);
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1 && work(batch, BTC_WORK_RELEASE, 1).id.operation == next_release.operation,
          "duplicate success for an older generation does not release its replacement");
    cache.receipt(next_release, 0);
    cache.pollMatch(1, BTC_PENDING);
    batch = cache.poll();
    const auto& restore = work(batch, BTC_WORK_DISK_TO_HOST, 1);
    check(restore.pages[0].source[0].pool == BTC_DISK_POOL && restore.pages[0].source[0].slot == disk_page.slot
              && restore.pages[0].source[0].generation == disk_page.generation,
          "Host release receipts leave the separate Disk allocation intact");
    std::array<uint8_t, 4> target{};
    buffer = {target.data(), target.size()};
    check(btc_execute_disk(cache.handle, restore.id, &buffer, 1) == BTC_OK && target == source,
          "read physical Disk page into a different Host generation");
    cache.finish(restore);
    const auto ready = cache.pollMatch(1);
    check(btc_plan_get(ready.get()).recoverable_tokens == 8, "Disk restore publishes usable Host");
    cache.close();
}

void disk_replica_survives_host_pressure() {
    const uint32_t owner[]{0};
    const std::string directory = std::filesystem::temp_directory_path().string();
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 2,
                {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_DEVICE_TO_HOST, owner, 1},
                 {BTC_WORK_HOST_TO_DISK, owner, 1}, {BTC_WORK_DISK_TO_HOST, owner, 1}},
                directory, true);
    const Keys keys{"retained-disk"};
    const btc_position position{0, 8};
    const btc_page initial_device{4, 9, 1, 1};
    cache.publish(keys, {{position, &initial_device, 1}});
    cache.demote(initial_device);
    cache.finish(work(cache.poll(), BTC_WORK_DEVICE_TO_HOST, 1));
    cache.finish_releases();
    check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 1) == BTC_OK, "spill first Host copy");
    auto batch = cache.poll();
    const auto& write = work(batch, BTC_WORK_HOST_TO_DISK, 1);
    const auto disk = write.pages[0].target[0];
    std::array<uint8_t, 4> payload{3, 0, 8, 9};
    btc_buffer buffer{payload.data(), payload.size()};
    check(btc_execute_disk(cache.handle, write.id, &buffer, 1) == BTC_OK, "write initial Disk bytes");
    cache.finish(write);
    cache.finish_releases();

    for (uint64_t round = 1; round <= 2; ++round) {
        cache.match(2 * round - 1, keys, 8, BTC_PENDING);
        batch = cache.poll();
        const auto& restore = work(batch, BTC_WORK_DISK_TO_HOST, 1);
        const auto& source = restore.pages[0].source[0];
        const auto host = restore.pages[0].target[0];
        check(source.pool == disk.pool && source.slot == disk.slot && source.generation == disk.generation,
              "restore uses the original Disk allocation");
        std::array<uint8_t, 4> bytes{};
        buffer = {bytes.data(), bytes.size()};
        check(btc_execute_disk(cache.handle, restore.id, &buffer, 1) == BTC_OK && bytes == payload,
              "Disk payload survives repeated restoration");
        cache.finish(restore);

        auto plan = cache.pollMatch(2 * round - 1);
        const btc_page target{4, 9, round + 1, round + 1};
        const btc_binding binding{position, &target, 1};
        check(btc_bind(cache.handle, plan.get(), &binding, 1) == BTC_WAITING, "bind restored Host to Device");
        cache.finish(work(cache.poll(), BTC_WORK_HOST_TO_DEVICE, 1));
        batch = cache.poll();
        check(btc_batch_get(batch.get()).work_count == 1, "successful Device load retires the staging Host row");
        const auto& release = work(batch, BTC_WORK_RELEASE, 1);
        const auto& released = release.pages[0].source[0];
        check(released.pool == BTC_HOST_POOL && released.slot == host.slot && released.generation == host.generation,
              "only the current Host allocation is released");
        cache.finish(release);

        auto ready = cache.match(2 * round, keys, 8);
        check(btc_plan_get(ready.get()).ready_tokens == 8 && btc_bind(cache.handle, ready.get(), nullptr, 0) == BTC_OK,
              "Device remains reusable after the Host slot is released");
        cache.invalidate(target);
        cache.finish_releases();
    }
    check(btc_reclaim_local(cache.handle, BTC_DISK_POOL, 0, 2) == BTC_OK, "reclaim all remaining Disk pages");
    batch = cache.poll();
    const auto& release = work(batch, BTC_WORK_RELEASE, 1);
    const auto& released = release.pages[0].source[0];
    check(released.pool == disk.pool && released.slot == disk.slot && released.generation == disk.generation,
          "the original Disk allocation is released exactly once");
    cache.finish(release);
    cache.close();
}

void disk_restore_respects_host_capacity() {
    const uint32_t owner[]{0};
    const auto directory = std::filesystem::temp_directory_path().string();
    for (const btc_rule rule : {BTC_FULL, BTC_LINEAR, BTC_SWA}) {
        const uint64_t window = rule == BTC_SWA ? 8 : 0;
        Cache cache({{0, rule, 8, window, 1, nullptr, 0, 0}}, 1, 2,
                    {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_DEVICE_TO_HOST, owner, 1},
                     {BTC_WORK_HOST_TO_DISK, owner, 1}, {BTC_WORK_DISK_TO_HOST, owner, 1}},
                    directory, true);
        const Keys first{"first"};
        const Keys both{"first", "second"};
        const btc_page device[]{{7, 8, 1, 1}, {7, 9, 1, 2}};
        const std::array<std::array<uint8_t, 4>, 2> values{{{{7, 0, 3, 5}}, {{9, 0, 4, 7}}}};
        for (size_t i = 0; i < 2; ++i) {
            cache.publish(i == 0 ? first : both, {{{0, 8 * (i + 1)}, &device[i], 1}});
            cache.demote(device[i]);
            cache.finish(work(cache.poll(), BTC_WORK_DEVICE_TO_HOST, 1));
            cache.finish_releases();
            check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 1) == BTC_OK, "fill one Disk slot");
            const auto batch = cache.poll();
            const auto& write = work(batch, BTC_WORK_HOST_TO_DISK, 1);
            auto payload = values[i];
            btc_buffer buffer{payload.data(), payload.size()};
            check(btc_execute_disk(cache.handle, write.id, &buffer, 1) == BTC_OK, "write each available Disk row");
            cache.finish(write);
            cache.finish_releases();
        }

        if (rule == BTC_FULL) {
            const uint64_t boundary = 16;
            const btc_limits only_full_workspace{16, nullptr, 0, &boundary, 1, SIZE_MAX, 0};
            Plan plan;
            check(CacheRef(cache.handle).match(1, both.get(), only_full_workspace, plan) == BTC_OK
                      && btc_plan_get(plan.get()).recoverable_tokens == 0,
                  "capacity cannot select an omitted endpoint");
        }
        cache.match(2, both, 16, BTC_PENDING);
        cache.match(3, both, 16, BTC_PENDING);
        const auto batch = cache.poll();
        check(btc_batch_get(batch.get()).work_count == 1, "requests share the one fitting Disk restore");
        const auto& read = work(batch, BTC_WORK_DISK_TO_HOST, 1);
        const size_t row = rule == BTC_FULL ? 0 : 1;
        const uint64_t endpoint = 8 * (row + 1);
        check(read.pages[0].position.end_token == endpoint, "native group rule chooses the fitting prefix");
        std::array<uint8_t, 4> restored{};
        btc_buffer buffer{restored.data(), restored.size()};
        check(btc_execute_disk(cache.handle, read.id, &buffer, 1) == BTC_OK && restored == values[row],
              "read only the required Disk row into Host");
        cache.finish(read);
        const auto events = cache.poll();
        check(has_event(events, 2, BTC_EVENT_PROGRESS) && has_event(events, 3, BTC_EVENT_PROGRESS),
              "both waiters can finish polling after the shared restore");
        for (uint64_t request : {2, 3}) {
            const auto plan = cache.pollMatch(request);
            const auto view = btc_plan_get(plan.get());
            check(view.ready_tokens == 0 && view.recoverable_tokens == endpoint && view.page_count == 1
                      && view.pages[0].route == BTC_WORK_HOST_TO_DEVICE,
                  "restore returns the largest legal endpoint that fits Host capacity");
        }
        cache.close();
    }
}

void disk_restore_reclaims_shared_host_capacity() {
    const uint32_t owner[]{0};
    const auto directory = std::filesystem::temp_directory_path().string();
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}, {1, BTC_LINEAR, 16, 0, 1, nullptr, 0, 0}}, 2, 3,
                {{BTC_WORK_HOST_TO_DEVICE, owner, 1}, {BTC_WORK_DEVICE_TO_HOST, owner, 1},
                 {BTC_WORK_HOST_TO_DISK, owner, 1}, {BTC_WORK_DISK_TO_HOST, owner, 1}},
                directory, true);
    const Keys first{"first"};
    const Keys both{"first", "second"};
    const btc_position positions[]{{0, 8}, {0, 16}, {1, 16}};
    const btc_page devices[]{{4, 9, 1, 1}, {4, 10, 1, 2}, {4, 11, 1, 3}};
    for (size_t i = 0; i < 3; ++i) {
        cache.publish(i == 0 ? first : both, {{positions[i], &devices[i], 1}});
        check(btc_demote(cache.handle, 4, 1, &devices[i], 1, 0) == BTC_OK, "copy current group to shared Host");
        cache.finish(work(cache.poll(), BTC_WORK_DEVICE_TO_HOST, 1));
        check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, positions[i].group, 1) == BTC_OK,
              "store each logical row on Disk");
        const auto batch = cache.poll();
        const auto& spill = work(batch, BTC_WORK_HOST_TO_DISK, 1);
        std::array<uint8_t, 4> payload{3, 0, 8, 1};
        btc_buffer buffer{payload.data(), payload.size()};
        check(btc_execute_disk(cache.handle, spill.id, &buffer, 1) == BTC_OK, "write a shared-pool Disk row");
        cache.finish(spill);
        cache.finish_releases();
    }
    check(btc_invalidate(cache.handle, devices, 3) == BTC_OK, "leave three Disk pages across two groups");
    cache.finish_releases();
    const auto overflow = cache.match(1, both, 16);
    check(btc_plan_get(overflow.get()).recoverable_tokens == 0, "three rows cannot share two slots");

    const auto match_group = [&](uint64_t request, uint32_t group, btc_status expected) {
        const btc_limits limits{16, &group, 1, nullptr, 0, SIZE_MAX, 0};
        Plan plan;
        check(CacheRef(cache.handle).match(request, both.get(), limits, plan) == expected, "select one group");
        return plan;
    };
    const auto finish_restore = [&](size_t rows, uint32_t group) {
        const auto batch = cache.poll();
        const auto& restore = work(batch, BTC_WORK_DISK_TO_HOST, rows);
        std::array<std::array<uint8_t, 4>, 2> payloads{};
        btc_buffer buffers[]{{payloads[0].data(), 4}, {payloads[1].data(), 4}};
        check(btc_execute_disk(cache.handle, restore.id, buffers, rows) == BTC_OK, "restore into the shared Host pool");
        for (size_t i = 0; i < rows; ++i) {
            check(restore.pages[i].position.group == group && payloads[i] == std::array<uint8_t, 4>{3, 0, 8, 1},
                  "restore bytes for the selected group");
        }
        cache.finish(restore);
    };
    match_group(2, 0, BTC_PENDING);
    finish_restore(2, 0);
    check(btc_plan_get(cache.pollMatch(2).get()).recoverable_tokens == 16, "first group fills Host");
    match_group(3, 1, BTC_PENDING);
    const auto batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1, "second group reclaims one shared Host slot");
    const auto& release = work(batch, BTC_WORK_RELEASE, 1);
    check(release.pages[0].source[0].pool == BTC_HOST_POOL, "shared capacity wakes after Host release");
    cache.finish(release);
    cache.pollMatch(3, BTC_PENDING);
    finish_restore(1, 1);
    check(btc_plan_get(cache.pollMatch(3).get()).recoverable_tokens == 16, "peer group uses reclaimed slot");
    cache.close();
}

void passive_status_does_not_dispatch() {
    const uint32_t ranks[]{0};
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 0,
                {{BTC_WORK_HOST_TO_DEVICE, ranks, 1}, {BTC_WORK_DEVICE_TO_HOST, ranks, 1}});
    const Keys keys{"queued"};
    const btc_page page{1, 1, 1, 1};
    cache.publish(keys, {{{0, 8}, &page, 1}});
    size_t count = 0;
    check(btc_pending_work(cache.handle, &count) == BTC_OK && count == 0, "Device publication queues no copy");
    cache.demote(page);
    check(btc_pending_work(cache.handle, &count) == BTC_OK && count == 1, "count queued write");
    check(btc_drain(cache.handle) == BTC_PENDING, "unsubmitted write can be cancelled during shutdown");
    auto batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1, "queued write was never exposed");
    cache.finish(work(batch, BTC_WORK_RELEASE, 1));
    cache.close();
}

void direct_remote_binding() {
    const uint32_t owner[]{0};
    const uint64_t pools[]{4, 5};
    Cache cache({{0, BTC_LINEAR, 8, 0, 2, pools, 0, 0}}, 0, 0,
                {{BTC_WORK_REMOTE_QUERY, owner, 1}, {BTC_WORK_REMOTE_TO_DEVICE, owner, 1},
                 {BTC_WORK_DEVICE_TO_REMOTE, owner, 1}});
    const Keys keys{"earlier-state", "checkpoint"};
    const btc_page committed[]{{4, 9, 1, 1}, {5, 9, 1, 2}};
    cache.publish(keys, {{{0, 16}, committed, 2}});
    auto batch = cache.poll();
    const auto write = work(batch, BTC_WORK_DEVICE_TO_REMOTE, 1).id;
    cache.invalidate(committed[0]);
    cache.receipt(write, 0, BTC_TIMEOUT, false);
    batch = cache.poll();
    check(btc_batch_get(batch.get()).work_count == 1, "unsafe direct remote write protects both device members");
    cache.receipt(write, 0, BTC_FAILED, true);
    batch = cache.poll();
    cache.finish(work(batch, BTC_WORK_RELEASE, 2));

    cache.match(1, keys, 16, BTC_PENDING, true);
    batch = cache.poll();
    const auto query = work(batch, BTC_WORK_REMOTE_QUERY, 2).id;
    const btc_remote_object object{{0, 16}, bytes("checkpoint-uri")};
    cache.receipt(query, 0, BTC_SUCCESS, true, false, &object, 1);
    const auto plan = cache.pollMatch(1);
    const auto view = btc_plan_get(plan.get());
    check(view.ready_tokens == 0 && view.host_tokens == 0 && view.recoverable_tokens == 16
              && view.page_count == 1 && view.pages[0].route == BTC_WORK_REMOTE_TO_DEVICE,
          "direct remote can recover a late LINEAR state without Host");
    const btc_page targets[]{{4, 9, 2, 3}, {5, 9, 2, 4}};
    const btc_binding binding{{0, 16}, targets, 2};
    check(btc_bind(cache.handle, plan.get(), &binding, 1) == BTC_WAITING, "bind direct grouped Device targets");
    batch = cache.poll();
    const auto& copy = work(batch, BTC_WORK_REMOTE_TO_DEVICE, 1);
    check(copy.pages[0].target_count == 2 && copy.pages[0].remote.size == object.descriptor.size,
          "direct remote operation receives grouped targets and opaque metadata");
    cache.finish(copy);
    check(has_event(cache.poll(), 1, BTC_EVENT_READY), "direct remote load wakes its request");
    const auto warm = cache.match(2, keys, 16, BTC_OK, true);
    check(btc_plan_get(warm.get()).ready_tokens == 16,
          "a full LINEAR checkpoint hit does not query earlier missing checkpoints");
    cache.close();
}

void scan_live_full_edges() {
    Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}});
    const Keys keys{"parent", "missing", "tail"};
    const btc_page page{0, 31, 1, 1};
    cache.publish(keys, {{{0, 24}, &page, 1}});
    check(btc_enable_prefix_scan(cache.handle) == BTC_OK, "enable scanner on live catalog");
    size_t size = 0, visited = 0, issues = 0;
    const auto visit = [](void* ctx, uint32_t group, btc_node_state parent, btc_node_state child) {
        check(group == 0 && !parent.tiers && child.tiers == BTC_ON_DEVICE, "gap to a device page");
        check(std::string_view(static_cast<const char*>(parent.key.data), parent.key.size) == "missing",
              "full edge uses preceding page");
        ++*static_cast<size_t*>(ctx);
    };
    for (size_t i = 0; i < 3; ++i) {
        check(btc_scan_prefix(cache.handle, i, 3, 1, &size, &visited, visit, &issues) == BTC_OK,
              "read bounded prefix scan");
        check(size == 3 && visited == 1, "scan only one node");
    }
    check(issues == 1, "report a missing parent once");
    cache.invalidate(page);
    cache.finish_releases();
    check(btc_scan_prefix(cache.handle, 0, 3, 1, &size, &visited, visit, &issues) == BTC_OK
              && !size && !visited, "scan index released the retired path");
    cache.close();
}

void local_eviction_policies() {
    const uint32_t rank = 0;
    for (const auto pool : {BTC_HOST_POOL, BTC_DISK_POOL}) {
        const bool disk = pool == BTC_DISK_POOL;
        for (const auto policy : {BTC_EVICT_FIFO, BTC_EVICT_LFU}) {
            std::vector<btc_route> routes{{BTC_WORK_HOST_TO_DEVICE, &rank, 1},
                                          {BTC_WORK_DEVICE_TO_HOST, &rank, 1}};
            if (disk) {
                routes.push_back({BTC_WORK_HOST_TO_DISK, &rank, 1});
                routes.push_back({BTC_WORK_DISK_TO_DEVICE, &rank, 1});
            }
            Cache cache({{0, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 2, disk ? 2 : 0,
                        routes, {}, false, disk);
            const Keys first{"first"};
            const Keys second{"second"};
            const btc_page a{1, 1, 1, 1};
            const btc_page b{1, 2, 1, 2};
            cache.publish(first, {{{0, 8}, &a, 1}});
            cache.demote(a, false);
            cache.finish(work(cache.poll(), BTC_WORK_DEVICE_TO_HOST, 1));
            cache.publish(second, {{{0, 8}, &b, 1}});
            cache.demote(b, false);
            cache.finish(work(cache.poll(), BTC_WORK_DEVICE_TO_HOST, 1));
            if (disk) {
                check(btc_reclaim_local(cache.handle, BTC_HOST_POOL, 0, 2) == BTC_OK, "spill both Host pages");
                const auto batch = cache.poll();
                const auto view = btc_batch_get(batch.get());
                std::multiset<std::string> disk_keys;
                std::set<uint64_t> disk_slots;
                for (size_t i = 0; i < view.work_count; ++i) {
                    check(view.work[i].kind == BTC_WORK_HOST_TO_DISK, "Disk write route");
                    for (size_t j = 0; j < view.work[i].page_count; ++j) {
                        const auto& page = view.work[i].pages[j];
                        check(page.target_count == 1 && page.target[0].pool == BTC_DISK_POOL, "Disk destination");
                        disk_slots.insert(page.target[0].slot);
                        disk_keys.emplace(static_cast<const char*>(page.key.data), page.key.size);
                    }
                    cache.finish(view.work[i]);
                }
                check(disk_keys == std::multiset<std::string>{"first", "second"} && disk_slots.size() == 2,
                      "native batches write both published keys to distinct Disk slots");
            }
            // The first arrival becomes both more recent and more frequently hit.
            for (uint64_t request = 1; request <= 2; ++request) {
                auto plan = cache.match(request, first, 8);
                check(btc_bind(cache.handle, plan.get(), nullptr, 0) == BTC_OK, "touch first arrival");
            }
            check(btc_set_local_eviction_policy(cache.handle, pool, policy) == BTC_OK, "set local eviction policy");
            check(btc_reclaim_local(cache.handle, pool, 0, 1) == BTC_OK, "reclaim one local page");
            cache.invalidate(a);
            cache.invalidate(b);
            cache.finish_releases();
            const auto kept = cache.match(3, policy == BTC_EVICT_FIFO ? second : first, 8);
            const auto removed = cache.match(4, policy == BTC_EVICT_FIFO ? first : second, 8);
            check(btc_plan_get(kept.get()).recoverable_tokens == 8 && btc_plan_get(removed.get()).recoverable_tokens == 0,
                  "FIFO discards the first arrival and LFU retains the frequently accessed page");
            cache.close();
        }
    }
}

void external_host_regions_use_native_slots() {
    constexpr size_t stride = 16, payload = 5;
    const uint32_t rank = 0;
    std::array<uint8_t, 3 * stride> memory;
    memory.fill(0x7f);
    const btc_host_region region{{memory.data(), memory.size()}, stride, payload};
    const std::vector<btc_route> load_route{{BTC_WORK_HOST_TO_DEVICE, &rank, 1}};
    const auto view = [&](const Cache& cache, uint32_t group, uint64_t slot, size_t row) {
        btc_buffer buffer{};
        check(btc_host_slot_buffer(cache.handle, group, slot, &buffer) == BTC_OK, "resolve Host slot");
        check(buffer.data == memory.data() + row * stride && buffer.size == payload,
              "Host slot aliases exactly its caller-owned payload at the configured stride");
        return static_cast<uint8_t*>(buffer.data);
    };
    const auto invalidSlot = [](const Cache& cache, uint32_t group, uint64_t slot) {
        btc_buffer buffer{};
        check(btc_host_slot_buffer(cache.handle, group, slot, &buffer) == BTC_INVALID,
              "missing mapping or invalid group/slot is rejected");
        check(!buffer.data && !buffer.size, "invalid Host view does not expose storage");
    };
    {
        Cache shared({{17, BTC_FULL, 8, 0, 1, nullptr, 0, 0}, {4, BTC_FULL, 8, 0, 1, nullptr, 0, 0}},
                     3, 0, load_route, {}, false, false, &region);
        check(view(shared, 4, 0, 0) == view(shared, 17, 0, 0), "groups share the same physical slot");
        view(shared, 17, 2, 2);
        invalidSlot(shared, 17, 3);
        invalidSlot(shared, 17, UINT64_MAX);
        invalidSlot(shared, 7, 0);
        shared.close();
    }
    {
        Cache partitioned({{17, BTC_FULL, 8, 0, 1, nullptr, 2, 0}, {4, BTC_FULL, 8, 0, 1, nullptr, 1, 0}},
                          0, 0, load_route, {}, false, false, &region);
        view(partitioned, 4, 0, 0);
        view(partitioned, 17, 1, 1);
        view(partitioned, 17, 2, 2);
        invalidSlot(partitioned, 17, 0);
        invalidSlot(partitioned, 4, 1);
        partitioned.close();
    }
    {
        Cache delegated({{4, BTC_FULL, 8, 0, 1, nullptr, 0, 0}}, 1, 0, load_route);
        invalidSlot(delegated, 4, 0);
        delegated.close();
    }
    {
        auto routes = load_route;
        routes.push_back({BTC_WORK_DEVICE_TO_HOST, &rank, 1});
        Cache mapped({{4, BTC_FULL, 8, 0, 1, nullptr, 0, 0}},
                     1, 0, routes, {}, false, false, &region);
        const btc_page page{1, 11, 1, 1};
        const Keys keys{"external-host"};
        mapped.publish(keys, {{{4, 8}, &page, 1}});
        mapped.demote(page);
        auto posted = mapped.poll();
        const auto& copy = work(posted, BTC_WORK_DEVICE_TO_HOST, 1);
        check(copy.pages[0].target_count == 1, "native demotion owns one Host target");
        const auto target = copy.pages[0].target[0];
        check(target.pool == BTC_HOST_POOL && target.slot == 0 && target.generation,
              "native transfer owns the public first Host slot");
        std::fill_n(view(mapped, 4, target.slot, 0), payload, 0x20);
        mapped.finish(copy);
        mapped.finish_releases();
        auto match = mapped.match(1, keys, 8);
        check(btc_plan_get(match.get()).recoverable_tokens == 8, "native match reuses the mapped Host page");
        check(std::all_of(memory.begin(), memory.begin() + payload, [](auto byte) { return byte == 0x20; })
                  && std::all_of(memory.begin() + payload, memory.end(), [](auto byte) { return byte == 0x7f; }),
              "mapped copy writes only the slot payload and preserves the padding and other slots");
        mapped.close();
    }
    const btc_group group{4, BTC_FULL, 8, 0, 1, nullptr, 0, 0};
    const btc_config config{btc_abi_version(), bytes("invalid external host"), 8, &group, 1, 2, 0, {},
                            load_route.data(), load_route.size(), 0, BTC_EVICT_LRU, 0, 0};
    for (const auto& bad : {btc_host_region{{memory.data(), stride}, stride, payload},
                            btc_host_region{{memory.data(), memory.size()}, SIZE_MAX, payload},
                            btc_host_region{{memory.data(), memory.size()}, stride, stride + 1}}) {
        btc_cache* rejected = nullptr;
        check(btc_open_with_host_region(&config, &bad, &rejected) == BTC_INVALID && !rejected,
              "short, overflowing, and mis-sized mappings cannot enter the native cache");
    }
}

}  // namespace

int main() {
    scan_live_full_edges();
    geometry_and_holds();
    reentrant_catalog_with_aliased_pages();
    reclaim_uses_the_requested_physical_pool();
    reclaim_keeps_all_aliases_until_unpinned();
    reclaim_counts_all_returned_slots_in_one_pool();
    related_reclaim_honors_cross_pool_selection();
    publication_and_load_move_only_the_selected_group();
    shared_load_unsafe_failure();
    invalidate_inflight_target();
    remote_warm_and_shared_timeout();
    host_spill_failure_can_retry_after_access();
    partitioned_demotion_and_disk_round_trip();
    disk_replica_survives_host_pressure();
    disk_restore_respects_host_capacity();
    disk_restore_reclaims_shared_host_capacity();
    passive_status_does_not_dispatch();
    direct_remote_binding();
    local_eviction_policies();
    external_host_regions_use_native_slots();
}
