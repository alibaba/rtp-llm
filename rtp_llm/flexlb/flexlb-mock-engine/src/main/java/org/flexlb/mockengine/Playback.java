package org.flexlb.mockengine;

import java.util.AbstractList;
import java.util.List;
import java.util.Map;

/** Shared playback policy; token remapping is stable across requests and shards. */
final class Playback {
    final PlaybackControls controls;
    final int maxLaps;
    final String identity;
    final double retain;
    final long seed;
    final double burstFactor, burstPeriod, burstDuty, diurnalAmplitude, diurnalPeriod;

    Playback(Map<String, String> env, boolean loop, long duration) {
        maxLaps = Integer.parseInt(text(env,"MAX_LAPS", loop ? "0" : "1"));
        if (maxLaps < 0 || (maxLaps == 0 && duration <= 0))
            throw new IllegalArgumentException("loop playback requires max_laps or bounded duration");
        identity = text(env,"LAP_IDENTITY", "structural-relabel");
        if (!List.of("none", "structural-relabel", "partial").contains(identity))
            throw new IllegalArgumentException("invalid lap identity");
        controls = new PlaybackControls(env, identity);
        retain = number(env, "LAP_RETAIN_PROBABILITY", 0, 0, 1);
        seed = Long.parseLong(text(env,"PLAYBACK_SEED", "0"));
        burstFactor = number(env, "BURST_FACTOR", 1, 1, Double.MAX_VALUE);
        burstPeriod = number(env, "BURST_PERIOD_SECONDS", 10, 0.001, Double.MAX_VALUE);
        burstDuty = number(env, "BURST_DUTY", 0.1, 0.001, 0.999);
        diurnalAmplitude = number(env, "DIURNAL_AMPLITUDE", 0, 0, 0.99);
        diurnalPeriod = number(env, "DIURNAL_PERIOD_SECONDS", 86400, 0.001, Double.MAX_VALUE);
    }

    private static String text(Map<String,String> env, String key, String fallback) {
        String value=env.get(key);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static double number(Map<String,String> env, String key, double fallback, double low, double high) {
        double v = Double.parseDouble(text(env,key, Double.toString(fallback)));
        if (!Double.isFinite(v) || v < low || v > high) throw new IllegalArgumentException("invalid " + key);
        return v;
    }

    boolean more(int completedLap) { return maxLaps == 0 || completedLap + 1 < maxLaps; }

    Map<String,Object> manifest() {
        Map<String,Object> result = new java.util.LinkedHashMap<>(Map.ofEntries(Map.entry("max_laps",maxLaps),Map.entry("identity",identity),
            Map.entry("retain_probability",retain),Map.entry("seed",seed),
            Map.entry("burst_factor",burstFactor),Map.entry("burst_period_seconds",burstPeriod),
            Map.entry("burst_duty",burstDuty),Map.entry("diurnal_amplitude",diurnalAmplitude),
            Map.entry("diurnal_period_seconds",diurnalPeriod)));
        controls.snapshot(result);
        return result;
    }

    /** Disjoint token ranges per fresh lap; fail instead of silently colliding. */
    List<Integer> relabel(List<Integer> tokens, List<Long> originalKeys, int blockSize,
                          int lap, long tokenStride) {
        if (lap == 0 || identity.equals("none")) return tokens;
        int blocks = (tokens.size()-1)/blockSize+1;
        int[] generations = new int[blocks];
        for (int b=0;b<blocks;b++) {
            int generation = lap;
            if (identity.equals("partial")) {
                long key = b < originalKeys.size() ? originalKeys.get(b) : tokens.hashCode();
                if (controls.schedule != null) {
                    generation = unit(mix(key ^ seed)) < controls.retention(lap) ? 0 : lap;
                } else while (generation > 0 && unit(mix(key ^ seed ^ (generation * 0x9e3779b97f4a7c15L))) < retain)
                    generation--;
            }
            if ((generation+1L)*tokenStride-1 > Integer.MAX_VALUE)
                throw new IllegalArgumentException("lap token identity budget exhausted");
            generations[b] = generation;
        }
        return new AbstractList<>() {
            @Override public Integer get(int index) {
                java.util.Objects.checkIndex(index,tokens.size());
                return Math.toIntExact(tokens.get(index)+generations[index/blockSize]*tokenStride);
            }
            @Override public int size() { return tokens.size(); }
        };
    }

    private static long mix(long x) {
        x=(x^(x>>>30))*0xbf58476d1ce4e5b9L;
        x=(x^(x>>>27))*0x94d049bb133111ebL;
        return x^(x>>>31);
    }
    private static double unit(long x) { return (x>>>11)*0x1.0p-53; }

    /** Monotone integrated arrival intensity, independent of sender lateness.
     * Numerical integration is used only for optional modulation (10ms cells).
     * Unmodulated uniform/ramp use the existing exact inverse.
     */
    final class Clock {
        private double cursor, count, randomIntensity;
        private long randomIndex = -1;
        double due(long index, double qps, double ramp) {
            if (controls.curve != null || controls.arrival.equals("poisson")) {
                double intensity = index;
                if (controls.arrival.equals("poisson")) {
                    if (index < randomIndex) throw new IllegalArgumentException("arrival indices must increase");
                    while (randomIndex < index) {
                        randomIndex++;
                        double u = unit(mix(randomIndex + (seed ^ 0xd1b54a32d192ed03L) + 0x9e3779b97f4a7c15L));
                        randomIntensity += u == 0 ? 0x1.0p-53 : -Math.log1p(-u);
                    }
                    intensity = randomIntensity;
                }
                return controls.inverse(intensity/qps);
            }
            if (burstFactor == 1 && diurnalAmplitude == 0)
                return JavaLoadClient.uniformDueSeconds(index,qps,ramp);
            if (index < count) throw new IllegalArgumentException("arrival indices must increase");
            while (true) {
                double end=cursor+0.01;
                // Split at burst edges, so narrow bursts are never skipped.
                double phase=cursor%burstPeriod;
                double edge=phase < burstPeriod*burstDuty-1e-9 ? burstPeriod*burstDuty-phase : burstPeriod-phase;
                if (edge > 1e-9) end=Math.min(end,cursor+edge);
                double middle=(cursor+end)/2;
                double burst=(middle%burstPeriod < burstPeriod*burstDuty ? burstFactor : 1)
                    /(1+(burstFactor-1)*burstDuty);
                double wave=1+diurnalAmplitude*Math.sin(2*Math.PI*middle/diurnalPeriod);
                double rate=qps*burst*wave*(ramp>0 ? Math.min(1,middle/ramp) : 1);
                double next=count+(end-cursor)*rate;
                if (next >= index) return cursor+(index-count)/rate;
                cursor=end;count=next;
            }
        }
    }
}
