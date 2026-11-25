package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class TtftStageQuantilesTest {
    private static final long BASE = 1_700_000_040_000L;
    private static final long WINDOW = TtftStageQuantiles.WINDOW_MS;
    private final List<Report> reports = new ArrayList<>();
    private final FlexMonitor monitor = monitor();
    private final TtftStageReporter reporter = new TtftStageReporter(monitor);

    @Test
    void selectsWholeRequestsAtNearestRankAndPublishesWindowMarkerLast() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 100, BASE);
        for (int index = 100; index > 0; index--) {
            quantiles.add("a", "r" + index, stages(1000 + index, 100 - index), BASE + index);
        }
        quantiles.advance(BASE + WINDOW - 1);
        assertTrue(reports.isEmpty());
        quantiles.advance(BASE + WINDOW);
        assertQuantile("a", BASE, "p50", stages(1050, 50));
        assertQuantile("a", BASE, "p95", stages(1095, 5));
        assertQuantile("a", BASE, "p99", stages(1099, 1));
        assertEquals(100, value("samples", windowTags("a", BASE)));
        assertEquals(BASE + WINDOW, value("window.end.ms", windowTags("a", BASE)));
        assertEquals(TtftStageReporter.WINDOW_PREFIX + "window.end.ms", reports.getLast().name());
        assertEquals(30, reports.size());
        Map<String, String> meanTags = Map.of("deployment_name", "a", "slot", slot(BASE), "statistic", "mean");
        assertEquals(1050.5, value("total.us", meanTags));
        assertEquals(49.5, value("before.prefill.us", meanTags));
        for (int index = 0; index < 7; index++) {
            verify(monitor).register(TtftStageReporter.WINDOW_PREFIX + TtftStageReporter.METRICS[index],
                    FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
        }
    }

    @Test
    void freezesCopiesAndReportsEachClosedWindowOnce() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 10, BASE);
        long[] original = stages(100, 10);
        quantiles.add("a", "r", original, BASE + 1);
        original[0] = 999;
        quantiles.advance(BASE + WINDOW);
        assertQuantile("a", BASE, "p50", stages(100, 10));
        reports.clear();
        quantiles.advance(BASE + WINDOW + 1_000);
        assertTrue(reports.isEmpty());
        reports.clear();
        quantiles.advance(BASE + 2 * WINDOW);
        assertTrue(reports.isEmpty());
        quantiles.advance(BASE + 3 * WINDOW);
        assertTrue(reports.isEmpty());
    }

    @Test
    void alternatesSlotsAndTreatsTheBoundaryAsTheNextWindow() {
        assertEquals(20_000, WINDOW);
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 10, BASE);
        quantiles.add("a", "last", stages(100, 1), BASE + WINDOW - 1);
        quantiles.add("a", "next", stages(200, 2), BASE + WINDOW);
        quantiles.advance(BASE + WINDOW);
        assertQuantile("a", BASE, "p95", stages(100, 1));
        reports.clear();
        quantiles.advance(BASE + 2 * WINDOW);
        assertQuantile("a", BASE + WINDOW, "p95", stages(200, 2));
        assertEquals(1, value("samples", windowTags("a", BASE + WINDOW)));
    }

    @Test
    void skipsStartupAndLeadershipPartialWindowsWithoutWritingZeroes() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 10, BASE + 1);
        quantiles.add("a", "partial", stages(100, 1), BASE + 2);
        quantiles.advance(BASE + WINDOW);
        assertTrue(reports.isEmpty());
        quantiles.add("a", "complete", stages(200, 2), BASE + WINDOW + 1);
        quantiles.advance(BASE + 2 * WINDOW);
        assertQuantile("a", BASE + WINDOW, "p50", stages(200, 2));
        reports.clear();
        quantiles.reset(BASE + 2 * WINDOW + 1);
        quantiles.advance(BASE + 2 * WINDOW + 2);
        quantiles.add("a", "new-leader-partial", stages(300, 3), BASE + 2 * WINDOW + 3);
        quantiles.advance(BASE + 3 * WINDOW);
        assertTrue(reports.isEmpty());
    }

    @Test
    void clockJumpsDoNotReemitAStaleOrIncompleteWindow() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 10, BASE);
        quantiles.add("a", "old", stages(100, 1), BASE + 1);
        quantiles.advance(BASE + 3 * WINDOW);
        assertTrue(reports.isEmpty());
        quantiles.add("a", "before-clock-correction", stages(200, 2), BASE + 3 * WINDOW + 1);
        quantiles.advance(BASE + 2 * WINDOW + 1);
        quantiles.advance(BASE + 3 * WINDOW);
        assertTrue(reports.isEmpty());
        quantiles.add("a", "clock-catching-up", stages(300, 3), BASE + 3 * WINDOW + 2);
        quantiles.advance(BASE + 4 * WINDOW);
        assertTrue(reports.isEmpty());
        quantiles.add("a", "clock-recovered", stages(400, 4), BASE + 4 * WINDOW + 1);
        quantiles.advance(BASE + 5 * WINDOW);
        assertQuantile("a", BASE + 4 * WINDOW, "p50", stages(400, 4));
    }

    @Test
    void overflowInvalidatesOnlyTheAffectedDeploymentAndRecoversNextWindow() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 3, BASE);
        quantiles.add("a", "a1", stages(100, 1), BASE + 1);
        quantiles.add("a", "a2", stages(200, 2), BASE + 2);
        quantiles.add("b", "b1", stages(300, 3), BASE + 3);
        quantiles.add("a", "a3", stages(400, 4), BASE + 4);
        quantiles.add("a", "a4", stages(500, 5), BASE + 5);
        quantiles.add("b", "b2", stages(600, 6), BASE + 6);
        quantiles.advance(BASE + WINDOW);
        assertEquals(4, value("samples", windowTags("a", BASE)));
        assertEquals(300, value("total.us", Map.of("deployment_name", "a", "slot", slot(BASE), "statistic", "mean")));
        assertTrue(reports.stream()
                .noneMatch(report -> "a".equals(report.tags().get("deployment_name"))
                        && "p50".equals(report.tags().get("statistic"))));
        assertEquals(2, value("samples", windowTags("b", BASE)));
        assertQuantile("b", BASE, "p95", stages(600, 6));
        reports.clear();
        quantiles.add("a", "recovered", stages(700, 7), BASE + WINDOW + 1);
        quantiles.advance(BASE + 2 * WINDOW);
        assertQuantile("a", BASE + WINDOW, "p50", stages(700, 7));
    }

    @Test
    void excludesPleOnlySamplesAndBreaksEqualTtftTiesByCompleteRequestId() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 10, BASE);
        quantiles.add("a", "ple-only", null, BASE + 1);
        quantiles.add("a", "b", stages(200, 20), BASE + 2);
        quantiles.add("a", "a", stages(200, 10), BASE + 3);
        quantiles.advance(BASE + WINDOW);
        assertEquals(2, value("samples", windowTags("a", BASE)));
        assertQuantile("a", BASE, "p50", stages(200, 10));
        assertQuantile("a", BASE, "p99", stages(200, 20));
    }

    @Test
    void deploymentCapacityNeverAdmitsOnlyTheLaterPartOfAnUnknownDeployment() {
        TtftStageQuantiles quantiles = new TtftStageQuantiles(reporter, 2, BASE);
        quantiles.add("a", "a1", stages(100, 1), BASE + 1);
        quantiles.add("b", "b1", stages(200, 2), BASE + 2);
        quantiles.add("c", "c1", stages(300, 3), BASE + 3);
        quantiles.add("a", "a2", stages(400, 4), BASE + 4);
        quantiles.add("c", "c2", stages(500, 5), BASE + 5);
        quantiles.advance(BASE + WINDOW);
        assertTrue(reports.stream().noneMatch(report -> "c".equals(report.tags().get("deployment_name"))));
        assertEquals(2, value("samples", windowTags("a", BASE)));
        assertEquals(1, value("samples", windowTags("b", BASE)));
    }

    private FlexMonitor monitor() {
        FlexMonitor result = mock(FlexMonitor.class);
        doAnswer(invocation -> {
            reports.add(new Report(invocation.getArgument(0), ((FlexMetricTags) invocation.getArgument(1)).getTags(),
                    invocation.getArgument(2)));
            return null;
        }).when(result).report(anyString(), any(FlexMetricTags.class), anyDouble());
        return result;
    }

    private void assertQuantile(String deployment, long startMs, String quantile, long[] stages) {
        Map<String, String> tags = Map.of("deployment_name", deployment, "slot", slot(startMs), "statistic", quantile);
        for (int index = 0; index < stages.length; index++) {
            assertEquals(stages[index], value(TtftStageReporter.METRICS[index], tags));
        }
    }

    private double value(String name, Map<String, String> tags) {
        return reports.stream()
                .filter(report -> report.name().equals(TtftStageReporter.WINDOW_PREFIX + name)
                        && report.tags().equals(tags))
                .findFirst()
                .orElseThrow()
                .value();
    }

    private static long[] stages(long total, long beforePrefill) {
        return new long[]{total, beforePrefill, 0, 0, 0, 0, total - beforePrefill};
    }

    private static Map<String, String> windowTags(String deployment, long startMs) {
        return Map.of("deployment_name", deployment, "slot", slot(startMs));
    }

    private static String slot(long startMs) {
        return Long.toString(startMs / WINDOW % 2);
    }

    private record Report(String name, Map<String, String> tags, double value) {
    }
}
