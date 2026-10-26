package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Request inactivity and post-decision Engine visibility are separate deadlines. */
@Getter
@Setter
public final class RequestLifecycleConfig {

    private RequestConfig request = new RequestConfig();
    private DecisionConfig decision = new DecisionConfig();

    @Getter
    @Setter
    public static final class RequestConfig {
        /** Maximum silence since registration or the latest matching Engine request status. */
        private Long timeoutMs;
        /**
         * Milliseconds allowed without matching request status after publication. V2 refreshes the
         * deadline on each waiting/running report; finished releases the request immediately.
         */
        private Long statusUpdateTimeoutMs = 1_000L;
    }

    @Getter
    @Setter
    public static final class DecisionConfig {
        private Double lifetime = 2.0;
    }
}
