package org.flexlb.cache.core;

import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link KvCacheManager#findMatchingRanksInGroup} — the per-rank
 * counterpart to {@code findMatchingEngines}. Other paths in KvCacheManager
 * (updateEngineCache / findMatchingEngines) are exercised via integration
 * with GlobalCacheIndexTest and the integration suite.
 */
@ExtendWith(MockitoExtension.class)
class KvCacheManagerTest {

    private EngineLocalView engineLocalView;
    private GlobalCacheIndex globalCacheIndex;
    private WorkerStatusProvider workerStatusProvider;
    private KvCacheManager kvCacheManager;

    @BeforeEach
    void setUp() throws Exception {
        engineLocalView = new EngineLocalView();
        injectField(engineLocalView, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
        injectField(engineLocalView, "dynamicIntervalManager",
                mock(org.flexlb.cache.service.DynamicCacheIntervalService.class));

        globalCacheIndex = new GlobalCacheIndex();
        workerStatusProvider = mock(WorkerStatusProvider.class);

        kvCacheManager = new KvCacheManager();
        injectField(kvCacheManager, "engineLocalView", engineLocalView);
        injectField(kvCacheManager, "globalCacheIndex", globalCacheIndex);
        injectField(kvCacheManager, "workerStatusProvider", workerStatusProvider);
        injectField(kvCacheManager, "cacheMetricsReporter", mock(CacheMetricsReporter.class));
    }

    @Test
    void per_rank_match_reports_distinct_lengths_across_ranks() {
        // Group has 4 ranks. Prefix = [1L, 2L, 3L, 4L].
        // R0 has [1,2,3,4]      → prefix length 4
        // R1 has [1,2]          → prefix length 2 (stops at 3)
        // R2 has []             → prefix length 0
        // R3 has [1,2,3,4,5]    → prefix length 4 (extra blocks don't help here)
        engineLocalView.replacePerRankBlocks("g1", List.of(
                setOf(1L, 2L, 3L, 4L),
                setOf(1L, 2L),
                Set.of(),
                setOf(1L, 2L, 3L, 4L, 5L)));

        Map<Integer, Integer> result = kvCacheManager.findMatchingRanksInGroup("g1",
                List.of(1L, 2L, 3L, 4L));

        assertEquals(4, result.size());
        assertEquals(4, result.get(0));
        assertEquals(2, result.get(1));
        assertEquals(0, result.get(2));
        assertEquals(4, result.get(3));
    }

    @Test
    void prefix_stops_at_first_miss_not_at_last_match() {
        // R0 has [1, 3, 4] but is MISSING 2. Prefix [1,2,3,4] should report 1
        // (matches block index 0 only — the gap at index 1 stops the prefix).
        engineLocalView.replacePerRankBlocks("g1", List.of(setOf(1L, 3L, 4L)));

        Map<Integer, Integer> result = kvCacheManager.findMatchingRanksInGroup("g1",
                List.of(1L, 2L, 3L, 4L));

        assertEquals(1, result.get(0),
                "prefix-match must stop at the first missing key, even if later keys are present");
    }

    @Test
    void unknown_group_returns_empty_map() {
        // No replacePerRankBlocks call ⇒ group not tracked.
        Map<Integer, Integer> result = kvCacheManager.findMatchingRanksInGroup("missing",
                List.of(1L, 2L));
        assertTrue(result.isEmpty(),
                "non-DP-enabled or unsynced group ⇒ empty map (caller falls back to group-level path)");
    }

    @Test
    void empty_or_null_inputs_return_empty_map() {
        engineLocalView.replacePerRankBlocks("g1", List.of(setOf(1L, 2L)));

        assertTrue(kvCacheManager.findMatchingRanksInGroup(null, List.of(1L, 2L)).isEmpty());
        assertTrue(kvCacheManager.findMatchingRanksInGroup("g1", null).isEmpty());
        assertTrue(kvCacheManager.findMatchingRanksInGroup("g1", List.of()).isEmpty());
    }

    @Test
    void result_includes_zero_match_ranks() {
        // Important: callers want to know 0-match ranks too (so they can route
        // AWAY from them). Ranks must NOT be omitted from the result.
        engineLocalView.replacePerRankBlocks("g1", List.of(setOf(99L), setOf(1L, 2L)));

        Map<Integer, Integer> result = kvCacheManager.findMatchingRanksInGroup("g1",
                List.of(1L, 2L));

        assertEquals(2, result.size());
        assertEquals(0, result.get(0), "zero-match rank must still appear in the result");
        assertEquals(2, result.get(1));
    }

    @Test
    void findMatchingEngines_returns_max_per_rank_for_DP_engines() {
        // Layout:
        //   engineA (DP=4): R0=[1,2], R1=[1,2,3,4], R2=[], R3=[1]
        //                   union has [1,2,3,4] → length 4
        //                   max-per-rank = max(2, 4, 0, 1) = 4
        //                   ⇒ same in this case (R1 already has the full prefix)
        //   engineB (DP=4): R0=[1], R1=[2], R2=[3], R3=[4]
        //                   union has [1,2,3,4] → length 4
        //                   max-per-rank = max(1, 1, 0, 0) = 1
        //                   ⇒ MAX strictly less than UNION here — this is the case
        //                     where the legacy union view over-estimated
        //   engineC (non-DP): cache = [1,2,3]
        //                     ⇒ scored from globalCacheIndex (legacy path) → length 3
        engineLocalView.replacePerRankBlocks("engineA", List.of(
                setOf(1L, 2L), setOf(1L, 2L, 3L, 4L), Set.of(), setOf(1L)));
        engineLocalView.replacePerRankBlocks("engineB", List.of(
                setOf(1L), setOf(2L), setOf(3L), setOf(4L)));
        // engineC: only feeds the global index (no per-rank → falls through)
        globalCacheIndex.addCacheBlock(1L, "engineC");
        globalCacheIndex.addCacheBlock(2L, "engineC");
        globalCacheIndex.addCacheBlock(3L, "engineC");

        when(workerStatusProvider.getWorkerIpPorts(any(), any()))
                .thenReturn(List.of("engineA", "engineB", "engineC"));

        Map<String, Integer> result = kvCacheManager.findMatchingEngines(
                List.of(1L, 2L, 3L, 4L), RoleType.PREFILL, null);

        assertEquals(3, result.size());
        assertEquals(4, result.get("engineA"), "DP engine where one rank has the full prefix ⇒ MAX = full prefix");
        assertEquals(1, result.get("engineB"),
                "DP engine where blocks are scattered across ranks ⇒ MAX < UNION (legacy view would have said 4)");
        assertEquals(3, result.get("engineC"), "non-DP engine ⇒ legacy union path unchanged");
    }

    @Test
    void findMatchingEngines_falls_back_to_global_index_when_perRank_not_synced() {
        // Engine X is DP-enabled in the deployment, but its dp_cache[] hasn't
        // been parsed yet (transient at startup) — falls back to global union.
        globalCacheIndex.addCacheBlock(1L, "engineX");
        globalCacheIndex.addCacheBlock(2L, "engineX");

        when(workerStatusProvider.getWorkerIpPorts(any(), any())).thenReturn(List.of("engineX"));

        Map<String, Integer> result = kvCacheManager.findMatchingEngines(
                List.of(1L, 2L), RoleType.PREFILL, null);

        assertEquals(2, result.get("engineX"), "no per-rank info ⇒ legacy path returns union prefix unchanged");
    }

    private static Set<Long> setOf(long... vals) {
        Set<Long> s = new HashSet<>();
        for (long v : vals) s.add(v);
        return s;
    }

    private static void injectField(Object target, String name, Object value) throws Exception {
        Field f = target.getClass().getDeclaredField(name);
        f.setAccessible(true);
        f.set(target, value);
    }
}
