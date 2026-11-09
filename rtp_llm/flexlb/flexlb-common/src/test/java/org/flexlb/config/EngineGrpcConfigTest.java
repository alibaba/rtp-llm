package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks the default KVCM receive limit, explicit overrides and rejection of nonpositive values.
 */
class EngineGrpcConfigTest {
    @Test
    void defaultsTo128MiB() {
        assertEquals(
                128 * 1024 * 1024,
                ConfigService.parse(document(""))
                        .getEngineGrpc()
                        .getKvCacheMaxInboundMessageBytes());
    }

    @Test
    void readsUnifiedConfigOverride() {
        assertEquals(
                67108864,
                ConfigService.parse(
                                document(
                                        ",\"engineGrpc\":{\"kvCacheMaxInboundMessageBytes\":67108864}"))
                        .getEngineGrpc()
                        .getKvCacheMaxInboundMessageBytes());
    }

    @Test
    void rejectsNonPositiveLimit() {
        for (int limit : new int[] {0, -1}) {
            assertThrows(
                    ConfigValidationException.class,
                    () ->
                            ConfigService.parse(
                                    document(
                                            ",\"engineGrpc\":{\"kvCacheMaxInboundMessageBytes\":"
                                                    + limit
                                                    + "}")));
        }
    }

    private static String document(String extra) {
        return "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":1000}}" + extra + "}";
    }
}
