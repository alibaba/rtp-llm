package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Auto-TPM SLO model: per-request SLO derived from a sequence-length bucket
 * base SLO multiplied by a priority multiplier.
 *
 * <p>Config formats (fall back to built-in defaults with a WARN on parse failure):
 * <ul>
 *   <li>{@code AUTO_TPM_SLO_LENGTH_BUCKETS=256:150,1024:300,4096:600,16384:1200,*:2400}
 *       — upper-bound-inclusive seqLen buckets to base SLO in ms; {@code *} is the
 *       catch-all bucket</li>
 *   <li>{@code AUTO_TPM_PRIORITY_SLO_MULTIPLIERS=30:2.0,40:1.5,50:1.0,60:0.75,70:0.5}
 *       — priority level to SLO multiplier</li>
 * </ul>
 */
@Slf4j
@Component
public class PrioritySloPolicy {

    public static final String DEFAULT_SLO_LENGTH_BUCKETS =
            "256:150,1024:300,4096:600,16384:1200,*:2400";
    public static final String DEFAULT_PRIORITY_SLO_MULTIPLIERS =
            "30:2.0,40:1.5,50:1.0,60:0.75,70:0.5";

    private static final String CATCH_ALL = "*";

    /** Sorted (bound asc) list of {bound, baseSloMs}; catch-all bound = Long.MAX_VALUE. */
    private final List<long[]> buckets;
    private final Map<Integer, Double> multipliers;

    @Autowired
    public PrioritySloPolicy(ConfigService configService) {
        this(configService.loadBalanceConfig().getAutoTpmSloLengthBuckets(),
                configService.loadBalanceConfig().getAutoTpmPrioritySloMultipliers());
    }

    public PrioritySloPolicy(String bucketSpec, String multiplierSpec) {
        this.buckets = parseBuckets(bucketSpec);
        this.multipliers = parseMultipliers(multiplierSpec);
    }

    /** Base SLO in ms for a sequence length (before priority multiplier). */
    public long baseSloMs(long seqLen) {
        for (long[] bucket : buckets) {
            if (seqLen <= bucket[0]) {
                return bucket[1];
            }
        }
        return buckets.get(buckets.size() - 1)[1];
    }

    /** SLO multiplier for a priority level; 1.0 for unknown levels. */
    public double multiplier(int priority) {
        return multipliers.getOrDefault(priority, 1.0);
    }

    /** Per-request SLO: baseSloMs(seqLen) * multiplier(priority). */
    public long requestSloMs(long seqLen, int priority) {
        return Math.round(baseSloMs(seqLen) * multiplier(priority));
    }

    /**
     * Latest admission deadline: the request must start prefill by this time
     * to have a chance of meeting its SLO.
     */
    public static long deadlineMs(long arrivalTimeMs, long requestSloMs, long predictedPrefillMs) {
        return arrivalTimeMs + requestSloMs - predictedPrefillMs;
    }

    /** Metric tag label of the bucket that a sequence length falls into. */
    public String bucketLabel(long seqLen) {
        for (long[] bucket : buckets) {
            if (seqLen <= bucket[0]) {
                return bucket[0] == Long.MAX_VALUE ? CATCH_ALL : String.valueOf(bucket[0]);
            }
        }
        return CATCH_ALL;
    }

    private static List<long[]> parseBuckets(String spec) {
        List<long[]> parsed = doParseBuckets(spec);
        if (parsed == null) {
            log.warn("Invalid autoTpmSloLengthBuckets '{}', falling back to default '{}'",
                    spec, DEFAULT_SLO_LENGTH_BUCKETS);
            parsed = doParseBuckets(DEFAULT_SLO_LENGTH_BUCKETS);
        }
        return parsed;
    }

    /**
     * Strict startup validation (F4/P0-4, used by {@link ConfigService}): the
     * first invalid fragment of a bucket spec, or {@code null} when the spec
     * is valid. Blank means "use built-in default" and is valid. Reuses the
     * same parse logic as the lenient runtime fallback, which stays unchanged.
     */
    static String firstInvalidBucketEntry(String spec) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        if (doParseBuckets(spec) != null) {
            return null;
        }
        for (String entry : spec.split(",")) {
            if (doParseBuckets(entry) == null) {
                return entry.trim();
            }
        }
        return spec;
    }

    private static List<long[]> doParseBuckets(String spec) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        List<long[]> result = new ArrayList<>();
        for (String entry : spec.split(",")) {
            String[] kv = entry.trim().split(":");
            if (kv.length != 2) {
                return null;
            }
            try {
                long bound = CATCH_ALL.equals(kv[0].trim())
                        ? Long.MAX_VALUE
                        : Long.parseLong(kv[0].trim());
                long sloMs = Long.parseLong(kv[1].trim());
                if (bound <= 0 || sloMs <= 0) {
                    return null;
                }
                result.add(new long[]{bound, sloMs});
            } catch (NumberFormatException e) {
                return null;
            }
        }
        if (result.isEmpty()) {
            return null;
        }
        result.sort(Comparator.comparingLong(a -> a[0]));
        return result;
    }

    private static Map<Integer, Double> parseMultipliers(String spec) {
        Map<Integer, Double> parsed = doParseMultipliers(spec);
        if (parsed == null) {
            log.warn("Invalid autoTpmPrioritySloMultipliers '{}', falling back to default '{}'",
                    spec, DEFAULT_PRIORITY_SLO_MULTIPLIERS);
            parsed = doParseMultipliers(DEFAULT_PRIORITY_SLO_MULTIPLIERS);
        }
        return parsed;
    }

    /**
     * Strict startup validation (F4/P0-4, used by {@link ConfigService}): the
     * first invalid fragment of a multiplier spec, or {@code null} when the
     * spec is valid. Blank means "use built-in default" and is valid.
     */
    static String firstInvalidMultiplierEntry(String spec) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        if (doParseMultipliers(spec) != null) {
            return null;
        }
        for (String entry : spec.split(",")) {
            if (doParseMultipliers(entry) == null) {
                return entry.trim();
            }
        }
        return spec;
    }

    private static Map<Integer, Double> doParseMultipliers(String spec) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        Map<Integer, Double> result = new HashMap<>();
        for (String entry : spec.split(",")) {
            String[] kv = entry.trim().split(":");
            if (kv.length != 2) {
                return null;
            }
            try {
                int priority = Integer.parseInt(kv[0].trim());
                double multiplier = Double.parseDouble(kv[1].trim());
                if (multiplier <= 0) {
                    return null;
                }
                result.put(priority, multiplier);
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return result.isEmpty() ? null : result;
    }
}
