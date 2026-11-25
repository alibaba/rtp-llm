package org.flexlb.service.monitor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Groups matched dispatches by deployment and collector time in 20-second windows.
 * At close, reports stage means and the entire dispatch at each nearest rank by total P TTFT.
 */
final class TtftStageQuantiles {
    static final long WINDOW_MS = 20_000;
    static final int[] PERCENTILES = {50, 95, 99};
    private final TtftStageReporter reporter;
    private final int maximumSamples;
    private final Map<String, DeploymentWindow> active = new LinkedHashMap<>();
    private long activeWindow;
    private long firstCompleteWindow;
    private int sampleCount;

    TtftStageQuantiles(TtftStageReporter reporter, int maximumEntries, long nowMs) {
        this.reporter = reporter;
        // Bound the exact sort independently of the larger request-correlation cache.
        this.maximumSamples = Math.min(maximumEntries, 65_536);
        reset(nowMs);
    }

    void add(String deployment, String requestId, long[] stages, long nowMs) {
        advance(nowMs);
        if (stages == null || activeWindow < firstCompleteWindow) {
            return;
        }
        DeploymentWindow window = active.get(deployment);
        if (window == null) {
            if (active.size() >= maximumSamples) {
                // Never admit part of a new deployment after discarding its earlier samples.
                return;
            }
            window = new DeploymentWindow();
            active.put(deployment, window);
        }
        window.count++;
        for (int stage = 0; stage < stages.length; stage++) {
            window.sums[stage] += stages[stage];
        }
        if (window.invalid) {
            return;
        }
        if (sampleCount >= maximumSamples) {
            // Only this deployment loses its percentile window; other retained cohorts stay exact.
            sampleCount -= window.samples.size();
            window.samples.clear();
            window.invalid = true;
            return;
        }
        window.samples.add(new Sample(requestId, stages.clone()));
        sampleCount++;
    }

    void reset(long nowMs) {
        active.clear();
        sampleCount = 0;
        activeWindow = Math.floorDiv(nowMs, WINDOW_MS);
        firstCompleteWindow = activeWindow + (Math.floorMod(nowMs, WINDOW_MS) == 0 ? 0 : 1);
        // Do not write zeroes from an old leader: it may share metric labels with the new leader.
    }

    void advance(long nowMs) {
        long windowIndex = Math.floorDiv(nowMs, WINDOW_MS);
        if (windowIndex < activeWindow) {
            // Keep the highest observed window so a wall-clock rollback cannot reuse a published window ID.
            active.clear();
            sampleCount = 0;
            firstCompleteWindow = Math.max(firstCompleteWindow, activeWindow + 1);
        } else if (windowIndex > activeWindow) {
            if (windowIndex == activeWindow + 1 && activeWindow >= firstCompleteWindow) {
                reportClosedWindow(windowIndex * WINDOW_MS);
            }
            active.clear();
            sampleCount = 0;
            activeWindow = windowIndex;
        }
    }

    private void reportClosedWindow(long endMs) {
        String slot = Long.toString(Math.floorMod(activeWindow, 2));
        for (var entry : active.entrySet()) {
            DeploymentWindow window = entry.getValue();
            double[] means = new double[window.sums.length];
            for (int stage = 0; stage < means.length; stage++) {
                means[stage] = window.sums[stage] / window.count;
            }
            reporter.reportWindowStages(entry.getKey(), slot, "mean", means);
            if (!window.invalid) {
                reportPercentiles(entry.getKey(), slot, window.samples);
            }
            reporter.reportWindowMetadata(entry.getKey(), slot, endMs, window.count);
        }
    }

    private void reportPercentiles(String deployment, String slot, List<Sample> samples) {
        samples.sort(Comparator.comparingLong((Sample sample) -> sample.stages()[0]).thenComparing(Sample::requestId));
        for (int percentile : PERCENTILES) {
            int rank = (int) (((long) samples.size() * percentile + 99) / 100) - 1;
            long[] stages = samples.get(rank).stages();
            reporter.reportWindowStages(deployment, slot, "p" + percentile,
                    Arrays.stream(stages).asDoubleStream().toArray());
        }
    }

    private static final class DeploymentWindow {
        private final List<Sample> samples = new ArrayList<>();
        private final double[] sums = new double[7];
        private long count;
        private boolean invalid;
    }

    private record Sample(String requestId, long[] stages) {
    }
}
