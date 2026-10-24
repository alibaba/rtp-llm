package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;

record MockCacheWatermarks(Tier device, Tier memory) {
    record Tier(double low, double high) {
        Tier {
            if (!(low == 0 && high == 0) && !(Double.isFinite(low) && Double.isFinite(high) && low > 0 && low < high && high <= 1))
                throw new IllegalArgumentException("cache watermark requires 0/0 or 0 < low < high <= 1");
        }
        boolean enabled() { return high > 0; }
        int lowBlocks(int total) { return (int)Math.floor(total * low); }
        int highBlocks(int total) { return (int)Math.ceil(total * high); }
    }
    static Tier parseTier(JsonNode config, String tier) {
        String prefix = "block_tree_" + tier + "_evict_";
        for (String key : new String[]{prefix + "low_watermark_ratio", prefix + "high_watermark_ratio"})
            if (!config.path(key).isNumber()) throw new IllegalArgumentException("missing numeric " + key);
        return new Tier(config.path(prefix + "low_watermark_ratio").doubleValue(), config.path(prefix + "high_watermark_ratio").doubleValue());
    }
    static MockCacheWatermarks parse(JsonNode config) { return new MockCacheWatermarks(parseTier(config, "device"), parseTier(config, "memory")); }
}
