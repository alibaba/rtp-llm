package org.flexlb.service.monitor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Groups finished engine tasks observed by this master in 20-second windows.
 * Each nonempty, complete window reports stage means. Within the sample limit, each percentile
 * reports the stage durations of the task at that nearest rank by total P engine TTFT.
 */
final class TtftStageQuantiles {
    static final long WINDOW_MS = 20_000;
    static final int[] PERCENTILES = {50, 95, 99};
    private final TtftStageReporter reporter;
    private final int maximumSamples;
    private final List<Sample> samples = new ArrayList<>();
    private final double[] sums = new double[TtftStageReporter.METRICS.length];
    private long activeWindow;
    private long firstCompleteWindow;
    private long count;

    TtftStageQuantiles(TtftStageReporter reporter, int maximumSamples, long nowMs) {
        this.reporter = reporter;
        this.maximumSamples = maximumSamples;
        reset(nowMs);
    }

    void add(String requestId, long[] stages, long nowMs) {
        advance(nowMs);
        if (activeWindow < firstCompleteWindow) {
            return;
        }
        count++;
        for (int stage = 0; stage < stages.length; stage++) {
            sums[stage] += stages[stage];
        }
        if (count <= maximumSamples) {
            samples.add(new Sample(requestId, stages));
        } else if (count == (long) maximumSamples + 1) {
            // The full cohort is needed for exact ranks. Continue collecting the mean.
            samples.clear();
        }
    }

    void reset(long nowMs) {
        clearSamples();
        activeWindow = Math.floorDiv(nowMs, WINDOW_MS);
        firstCompleteWindow = activeWindow + (Math.floorMod(nowMs, WINDOW_MS) == 0 ? 0 : 1);
        // Do not write zeroes from an old leader: it may share metric labels with the new leader.
    }

    void advance(long nowMs) {
        long windowIndex = Math.floorDiv(nowMs, WINDOW_MS);
        if (windowIndex < activeWindow) {
            // Keep the highest observed window so a wall-clock rollback cannot reuse a published window ID.
            clearSamples();
            firstCompleteWindow = Math.max(firstCompleteWindow, activeWindow + 1);
        } else if (windowIndex > activeWindow) {
            if (windowIndex == activeWindow + 1 && activeWindow >= firstCompleteWindow) {
                reportClosedWindow(windowIndex * WINDOW_MS);
            }
            clearSamples();
            activeWindow = windowIndex;
        }
    }

    private void clearSamples() {
        samples.clear();
        Arrays.fill(sums, 0);
        count = 0;
    }

    private void reportClosedWindow(long endMs) {
        if (count == 0) {
            return;
        }
        String slot = Long.toString(Math.floorMod(activeWindow, 2));
        double[] means = new double[sums.length];
        for (int stage = 0; stage < sums.length; stage++) {
            means[stage] = sums[stage] / count;
        }
        reporter.reportWindowStages(slot, "mean", means);
        if (!samples.isEmpty()) {
            reportPercentiles(slot);
        }
        reporter.reportWindowMetadata(slot, endMs, count);
    }

    private void reportPercentiles(String slot) {
        samples.sort(Comparator.comparingLong((Sample sample) -> sample.stages()[0]).thenComparing(Sample::requestId));
        for (int percentile : PERCENTILES) {
            int rank = (int) (((long) samples.size() * percentile + 99) / 100) - 1;
            long[] stages = samples.get(rank).stages();
            reporter.reportWindowStages(slot, "p" + percentile,
                    Arrays.stream(stages).asDoubleStream().toArray());
        }
    }

    private record Sample(String requestId, long[] stages) {
    }
}
