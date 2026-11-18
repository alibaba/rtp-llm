package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.util.Map;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.scheduler.LLMMockEngineTest;
import org.flexlb.mock.protocol.Role;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks that Remote Cache identity follows deployment and cache compatibility, not the Pod lifecycle.
 */
class KvcmRemoteCacheTest {
    @Test void replicasAndRestartsShareIdentityButDeploymentsAndLayoutsDoNot() throws Exception {
        var config = config();
        var env = Map.of("BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "dsv41", "HIPPO_POD_IP", "10.0.0.1");
        var first = KvcmRemoteCache.registration(config, env);
        assertEquals(first, KvcmRemoteCache.registration(config(), Map.of(
            "BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "dsv41", "HIPPO_POD_IP", "10.0.0.2")));
        assertNotEquals(first.path("instance_id"), KvcmRemoteCache.registration(config, Map.of(
            "BIZ_NAME", "mock-engine", "DEPLOYMENT_NAME", "other")).path("instance_id"));
        var changed = root();
        ((ObjectNode) changed.path("component").path("config")).put("prefill_cp_size", 4);
        assertNotEquals(first.path("instance_id"), KvcmRemoteCache.registration(RtpConfig.load(changed), env).path("instance_id"));
        assertEquals(config.tokensPerCacheKey(), first.path("block_size").asInt());
        assertEquals("default", first.path("instance_group").asText());
    }

    @Test void nativeGroupAndSaltAreAcceptedFromWhaleEnvironment() throws Exception {
        var config = RtpConfig.fromEnvironment("p-1", Role.prefill, Map.of(
            "TEST_BLOCK_NUM", "100", "MOCK_COMPUTE_FORWARD_EXPRESSION", "1", "RECO_INSTANCE_GROUP", "shared", "RECO_INSTANCE_ID_SALT", "scope"));
        var registration = KvcmRemoteCache.registration(config, Map.of());
        assertEquals("shared", registration.path("instance_group").asText());
        assertNotEquals(registration.path("instance_id"), KvcmRemoteCache.registration(config(), Map.of()).path("instance_id"));
    }

    private RtpConfig config() throws Exception { return RtpConfig.load(root()); }

    private ObjectNode root() {
        return new ObjectMapper().valueToTree(LLMMockEngineTest.configuration("prefill"));
    }
}
