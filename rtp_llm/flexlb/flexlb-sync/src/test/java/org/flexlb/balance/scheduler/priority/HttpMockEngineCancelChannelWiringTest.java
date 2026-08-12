package org.flexlb.balance.scheduler.priority;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Wiring tests for the test-only {@link HttpMockEngineCancelChannel}
 * (@ConditionalOnProperty hard red line: production wiring must be
 * byte-identical when the property is absent):
 * <ul>
 *   <li>property ABSENT → the bean is NOT created at all; the context wires
 *       exactly today's {@link UnsupportedEngineCancelChannel} and the
 *       {@code @Primary} never participates in resolution (no ambiguity, no
 *       behavior change),</li>
 *   <li>property SET (env {@code FLEXLB_TEST_MOCK_CANCEL_CONTROL_URL} via
 *       relaxed binding) → the bean exists and wins the
 *       {@link EngineCancelChannel} resolution as {@code @Primary},</li>
 *   <li>property set to the EMPTY STRING (third state, e.g. an exported but
 *       blank env var) → the bean IS created (@ConditionalOnProperty matches
 *       any non-"false" value) but {@code isSupported} reports false, so the
 *       channel degrades to unsupported instead of firing requests at an
 *       empty URL.</li>
 * </ul>
 */
class HttpMockEngineCancelChannelWiringTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withUserConfiguration(UnsupportedEngineCancelChannel.class,
                    HttpMockEngineCancelChannel.class);

    @Test
    void propertyAbsentBeanNotCreatedProductionWiringUnchanged() {
        runner.run(context -> {
            assertThat(context).doesNotHaveBean(HttpMockEngineCancelChannel.class);
            // Single candidate — resolution is exactly the production default.
            assertThat(context).hasSingleBean(EngineCancelChannel.class);
            assertThat(context.getBean(EngineCancelChannel.class))
                    .isInstanceOf(UnsupportedEngineCancelChannel.class);
        });
    }

    @Test
    void propertySetBeanCreatedAndPrimaryWinsResolution() {
        runner.withPropertyValues(
                        "flexlb.test.mock-cancel-control-url=http://127.0.0.1:18089")
                .run(context -> {
                    assertThat(context).hasSingleBean(HttpMockEngineCancelChannel.class);
                    // Both channels exist; @Primary resolves to the HTTP one.
                    assertThat(context.getBean(EngineCancelChannel.class))
                            .isInstanceOf(HttpMockEngineCancelChannel.class);
                });
    }

    @Test
    void configuredChannelSupportsEndpointsWithoutNetworkCalls() {
        runner.withPropertyValues(
                        "flexlb.test.mock-cancel-control-url=http://127.0.0.1:18089/")
                .run(context -> {
                    HttpMockEngineCancelChannel channel =
                            context.getBean(HttpMockEngineCancelChannel.class);
                    // isSupported is a pure URL-configured check (single mock
                    // control plane address covers the whole topology).
                    assertThat(channel.isSupported(null)).isTrue();
                });
    }

    @Test
    void emptyPropertyThirdStateBeanCreatedButUnsupported() {
        // Empty string is NOT "absent": @ConditionalOnProperty(matchIfMissing
        // = false) still matches any present non-"false" value, so the bean
        // is created — but the blank URL must degrade to isSupported=false
        // rather than issuing cancels against an empty base URL.
        runner.withPropertyValues("flexlb.test.mock-cancel-control-url=")
                .run(context -> {
                    assertThat(context).hasSingleBean(HttpMockEngineCancelChannel.class);
                    assertThat(context.getBean(HttpMockEngineCancelChannel.class)
                            .isSupported(null)).isFalse();
                });
    }
}
