package org.flexlb.service.monitor;

import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class TtftStageQuantilesTest {
    private static final long WINDOW = 20_000;

    @Test
    void reportsMeanAndStagesOfTheTaskAtEachTtftRank() {
        FlexMonitor monitor = mock(FlexMonitor.class);
        var windows = new TtftStageQuantiles(new TtftStageReporter(monitor), 100, 0);
        for (int rank = 100; rank >= 1; rank--) {
            windows.add("task-" + rank, stages(1000 + rank, 100 - rank), rank);
        }
        windows.advance(WINDOW);

        assertReported(monitor, "mean", 1050.5, 49.5);
        assertReported(monitor, "p50", 1050, 50);
        assertReported(monitor, "p95", 1095, 5);
        assertReported(monitor, "p99", 1099, 1);
        var tags = FlexMetricTags.of("slot", "0");
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "samples", tags, 100d);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "window.end.ms", tags, (double) WINDOW);
    }

    @Test
    void skipsPartialWindowAndBreaksTiesByTaskIdInNextWindow() {
        var reporter = mock(TtftStageReporter.class);
        var windows = new TtftStageQuantiles(reporter, 10, 1);
        windows.add("partial", stages(50, 5), 2);
        windows.add("b", stages(100, 20), WINDOW);
        windows.add("a", stages(100, 10), WINDOW + 1);
        verifyNoInteractions(reporter);

        windows.advance(2 * WINDOW);
        verify(reporter).reportWindowMetadata("1", 2 * WINDOW, 2);
        verify(reporter).reportWindowStages("1", "p50", new double[]{100, 10, 0, 0, 0, 90});
        verify(reporter).reportWindowStages("1", "p99", new double[]{100, 20, 0, 0, 0, 80});
    }

    @Test
    void overflowKeepsAllTasksInTheMeanAndSkipsPercentiles() {
        var reporter = mock(TtftStageReporter.class);
        var windows = new TtftStageQuantiles(reporter, 2, 0);
        windows.add("first", stages(100, 10), 1);
        windows.add("only", stages(200, 20), 2);
        windows.add("overflow", stages(300, 30), 3);
        windows.advance(WINDOW);

        verify(reporter).reportWindowStages("0", "mean", new double[]{200, 20, 0, 0, 0, 180});
        verify(reporter, never()).reportWindowStages(eq("0"), eq("p99"), any());
        verify(reporter).reportWindowMetadata("0", WINDOW, 3);
    }

    private static void assertReported(FlexMonitor monitor, String statistic, double total, double kvWait) {
        var tags = FlexMetricTags.of(Map.of("slot", "0", "statistic", statistic));
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "total.us", tags, total);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "kv.alloc.us", tags, kvWait);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "other.us", tags, total - kvWait);
    }

    private static long[] stages(long total, long kvWait) {
        return new long[]{total, kvWait, 0, 0, 0, total - kvWait};
    }
}
