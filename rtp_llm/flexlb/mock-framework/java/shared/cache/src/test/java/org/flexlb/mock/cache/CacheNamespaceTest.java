package org.flexlb.mock.cache;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CacheNamespaceTest {
    @Test void seedUsesOnlyTheSuppliedModelAndEnvironment() throws Exception {
        var json = new ObjectMapper();
        var model = json.readTree("{\"model_type\":\"deepseek_v41\",\"cache_key_hash_seed\":123}");
        assertEquals(0x4453563431494631L, CacheSequence.cacheKeySeed(model, Map.of()));
        assertEquals(0x4453563431494231L, CacheSequence.cacheKeySeed(model, Map.of("DSV41_SWA_BOUNDED_REPLAY", "1")));
        model = json.readTree("{\"cache_key_hash_seed\":123}");
        assertEquals(123, CacheSequence.cacheKeySeed(model, Map.of()));
        assertEquals(0x4453563431494631L, CacheSequence.cacheKeySeed(model, Map.of("MODEL_TYPE", "deepseek_v41")));
    }
}
