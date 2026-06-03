package org.flexlb.balance.strategy;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.util.JsonUtils;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * 2D grid lookup predictor: (log₂(ΣcomputeTokens), batchSize) → medianMs.
 *
 * <p>Each cell stores a median latency and a sample count. Cells with fewer than
 * {@code minSamples} observations are treated as empty. Lookup falls back to
 * nearest-neighbor interpolation within the same batch-size row, then to an
 * optional delegate predictor.
 */
public class GridPredictor implements PrefillTimePredictor {

    private final Map<Long, Long> grid;
    private final Map<Long, Integer> counts;
    private final int minSamples;
    private final PrefillTimePredictor fallback;

    public GridPredictor(Map<Long, Long> grid, Map<Long, Integer> counts,
                         int minSamples, PrefillTimePredictor fallback) {
        this.grid = grid;
        this.counts = counts;
        this.minSamples = minSamples;
        this.fallback = fallback;
    }

    /**
     * Parse a grid table from JSON.
     *
     * <pre>{@code
     * {
     *   "cells": {
     *     "13:1": {"ms": 180, "n": 50},
     *     "14:2": {"ms": 360, "n": 30}
     *   },
     *   "minSamples": 10
     * }
     * }</pre>
     *
     * Keys are {@code "log2Bin:bs"}.
     */
    public static GridPredictor fromJson(String json, PrefillTimePredictor fallback) {
        JsonNode root = JsonUtils.toTreeNode(json);
        int minSamples = root.path("minSamples").asInt(1);
        if (minSamples <= 0) {
            minSamples = 1;
        }
        JsonNode cells = root.path("cells");

        Map<Long, Long> grid = new HashMap<>();
        Map<Long, Integer> counts = new HashMap<>();

        if (cells.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = cells.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                String[] parts = entry.getKey().split(":");
                if (parts.length != 2) {
                    continue;
                }
                int log2Bin = Integer.parseInt(parts[0].trim());
                int bs = Integer.parseInt(parts[1].trim());
                long key = packKey(log2Bin, bs);

                JsonNode cell = entry.getValue();
                grid.put(key, cell.path("ms").asLong());
                counts.put(key, cell.path("n").asInt());
            }
        }

        return new GridPredictor(grid, counts, minSamples, fallback);
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        return predictBatchMs(List.of(new RequestProfile(totalTokens, hitTokens)));
    }

    @Override
    public long predictBatchMs(List<RequestProfile> requests) {
        if (requests.isEmpty()) {
            return 0;
        }
        int bs = requests.size();
        long sumC = 0;
        for (RequestProfile r : requests) {
            sumC += r.computeTokens();
        }
        int bin = log2Bin(Math.max(sumC, 1));

        Long ms = lookup(bin, bs);
        if (ms != null) {
            return ms;
        }

        ms = findNearest(bin, bs);
        if (ms != null) {
            return ms;
        }

        if (fallback != null) {
            return fallback.predictBatchMs(requests);
        }
        return 0;
    }

    private Long lookup(int bin, int bs) {
        long key = packKey(bin, bs);
        Long ms = grid.get(key);
        if (ms == null) {
            return null;
        }
        Integer n = counts.get(key);
        if (n != null && n < minSamples) {
            return null;
        }
        return ms;
    }

    private Long findNearest(int targetBin, int bs) {
        Long lower = null;
        int lowerBin = -1;
        Long upper = null;
        int upperBin = -1;

        for (int delta = 1; delta <= 4; delta++) {
            if (lower == null) {
                Long v = lookup(targetBin - delta, bs);
                if (v != null) {
                    lower = v;
                    lowerBin = targetBin - delta;
                }
            }
            if (upper == null) {
                Long v = lookup(targetBin + delta, bs);
                if (v != null) {
                    upper = v;
                    upperBin = targetBin + delta;
                }
            }
            if (lower != null && upper != null) {
                break;
            }
        }

        if (lower != null && upper != null) {
            double frac = (double) (targetBin - lowerBin) / (upperBin - lowerBin);
            return (long) (lower + frac * (upper - lower));
        }
        if (lower != null) {
            return lower;
        }
        return upper;
    }

    static long packKey(int log2Bin, int bs) {
        return ((long) log2Bin << 32) | (bs & 0xFFFFFFFFL);
    }

    static int log2Bin(long value) {
        if (value <= 0) {
            return 0;
        }
        return 63 - Long.numberOfLeadingZeros(value);
    }
}
