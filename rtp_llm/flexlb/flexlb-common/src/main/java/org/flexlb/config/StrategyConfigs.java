package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.Locale;

@Data
@Slf4j
@JsonIgnoreProperties(ignoreUnknown = true)
public class StrategyConfigs {

    private ShortestTtftStrategyConfig shortestTtft = new ShortestTtftStrategyConfig();

    public void normalize() {
        if (shortestTtft == null) {
            shortestTtft = new ShortestTtftStrategyConfig();
        }
        shortestTtft.normalize();
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class ShortestTtftStrategyConfig {

        private CandidatePoolConfig candidatePool = new CandidatePoolConfig();

        private void normalize() {
            if (candidatePool == null) {
                candidatePool = new CandidatePoolConfig();
            }
            candidatePool.normalize();
        }
    }

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class CandidatePoolConfig {

        public static final CandidatePoolMode DEFAULT_MODE = CandidatePoolMode.RATIO;
        public static final double DEFAULT_RATIO = 0.3;
        public static final int DEFAULT_MIN_SIZE = 1;
        public static final int DEFAULT_SIZE = 1;

        private CandidatePoolMode mode = DEFAULT_MODE;
        private double ratio = DEFAULT_RATIO;
        private int minSize = DEFAULT_MIN_SIZE;
        private int size = DEFAULT_SIZE;

        public int resolveCandidateCount(int workerCount) {
            if (workerCount <= 0) {
                return 0;
            }

            CandidatePoolMode resolvedMode = mode != null ? mode : DEFAULT_MODE;
            double resolvedRatio = isValidRatio(ratio) ? ratio : DEFAULT_RATIO;
            int resolvedMinSize = Math.max(DEFAULT_MIN_SIZE, minSize);
            int resolvedSize = Math.max(DEFAULT_SIZE, size);

            int candidateCount;
            if (resolvedMode == CandidatePoolMode.FIXED) {
                candidateCount = resolvedSize;
            } else {
                candidateCount = Math.max(resolvedMinSize, (int) (workerCount * resolvedRatio));
            }
            return Math.min(workerCount, Math.max(DEFAULT_MIN_SIZE, candidateCount));
        }

        private void normalize() {
            if (mode == null) {
                log.warn("Invalid shortestTtft candidatePool mode: null, fallback to default: {}", DEFAULT_MODE);
                mode = DEFAULT_MODE;
            }

            if (!isValidRatio(ratio)) {
                log.warn("Invalid shortestTtft candidatePool ratio: {}, fallback to default: {}", ratio, DEFAULT_RATIO);
                ratio = DEFAULT_RATIO;
            }

            if (minSize < DEFAULT_MIN_SIZE) {
                log.warn("Invalid shortestTtft candidatePool minSize: {}, fallback to default: {}", minSize, DEFAULT_MIN_SIZE);
                minSize = DEFAULT_MIN_SIZE;
            }

            if (size < DEFAULT_MIN_SIZE) {
                log.warn("Invalid shortestTtft candidatePool size: {}, fallback to default: {}", size, DEFAULT_SIZE);
                size = DEFAULT_SIZE;
            }
        }

        private boolean isValidRatio(double value) {
            return Double.isFinite(value) && value > 0.0 && value <= 1.0;
        }
    }

    public enum CandidatePoolMode {
        RATIO,
        FIXED;

        @JsonCreator
        public static CandidatePoolMode fromString(String value) {
            if (value == null || value.trim().isEmpty()) {
                return null;
            }
            try {
                return CandidatePoolMode.valueOf(value.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                return null;
            }
        }
    }
}
