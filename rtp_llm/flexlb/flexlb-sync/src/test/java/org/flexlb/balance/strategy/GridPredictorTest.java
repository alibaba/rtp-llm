package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GridPredictorTest {

    @Test
    void directHit() {
        // bin 13 = [8192, 16383], bs=1
        Map<Long, Long> grid = Map.of(GridPredictor.packKey(13, 1), 180L);
        Map<Long, Integer> counts = Map.of(GridPredictor.packKey(13, 1), 50);

        GridPredictor gp = new GridPredictor(grid, counts, 10, null);

        // 10000 tokens → log2Bin(10000) = 13
        long ms = gp.estimateMs(10000, 0);
        assertEquals(180, ms);
    }

    @Test
    void belowMinSamplesFallsBack() {
        Map<Long, Long> grid = Map.of(GridPredictor.packKey(13, 1), 180L);
        Map<Long, Integer> counts = Map.of(GridPredictor.packKey(13, 1), 5);

        PolynomialPredictor poly = new PolynomialPredictor(100, 0, 0, 0, 0, 0);
        GridPredictor gp = new GridPredictor(grid, counts, 10, poly);

        long ms = gp.estimateMs(10000, 0);
        assertEquals(100, ms);
    }

    @Test
    void nearestNeighborInterpolation() {
        // bin 12 → 200ms, bin 14 → 400ms, query bin 13
        long key12 = GridPredictor.packKey(12, 1);
        long key14 = GridPredictor.packKey(14, 1);
        Map<Long, Long> grid = Map.of(key12, 200L, key14, 400L);
        Map<Long, Integer> counts = Map.of(key12, 20, key14, 20);

        GridPredictor gp = new GridPredictor(grid, counts, 10, null);

        long ms = gp.estimateMs(10000, 0); // bin 13
        // linear interpolation: 200 + (13-12)/(14-12) * (400-200) = 300
        assertEquals(300, ms);
    }

    @Test
    void nearestNeighborOneSide() {
        long key12 = GridPredictor.packKey(12, 1);
        Map<Long, Long> grid = Map.of(key12, 200L);
        Map<Long, Integer> counts = Map.of(key12, 20);

        GridPredictor gp = new GridPredictor(grid, counts, 10, null);

        long ms = gp.estimateMs(10000, 0); // bin 13, only lower neighbor
        assertEquals(200, ms);
    }

    @Test
    void batchLookup() {
        // bs=2, sumC=16384 → bin 14
        long key = GridPredictor.packKey(14, 2);
        Map<Long, Long> grid = Map.of(key, 360L);
        Map<Long, Integer> counts = Map.of(key, 30);

        GridPredictor gp = new GridPredictor(grid, counts, 10, null);

        List<RequestProfile> batch = List.of(
                new RequestProfile(10000, 0),
                new RequestProfile(8000, 1616));
        // sumC = 10000 + 6384 = 16384, bin = 14, bs = 2
        long ms = gp.predictBatchMs(batch);
        assertEquals(360, ms);
    }

    @Test
    void fromJsonParsing() {
        String json =
                """
                {
                  "cells": {
                    "13:1": {"ms": 180, "n": 50},
                    "14:2": {"ms": 360, "n": 30}
                  },
                  "minSamples": 10
                }
                """;

        GridPredictor gp = GridPredictor.fromJson(json, null);

        assertEquals(180, gp.estimateMs(10000, 0));

        List<RequestProfile> batch = List.of(
                new RequestProfile(10000, 0),
                new RequestProfile(8000, 1616));
        assertEquals(360, gp.predictBatchMs(batch));
    }

    @Test
    void emptyBatchReturnsZero() {
        GridPredictor gp = GridPredictor.fromJson("{}", null);
        assertEquals(0, gp.predictBatchMs(List.of()));
    }

    @Test
    void noMatchNoFallbackReturnsZero() {
        GridPredictor gp = GridPredictor.fromJson("{}", null);
        assertEquals(0, gp.estimateMs(10000, 0));
    }

    @Test
    void log2BinValues() {
        assertEquals(0, GridPredictor.log2Bin(1));
        assertEquals(10, GridPredictor.log2Bin(1024));
        assertEquals(13, GridPredictor.log2Bin(8192));
        assertEquals(13, GridPredictor.log2Bin(10000));
        assertEquals(16, GridPredictor.log2Bin(65536));
    }
}
