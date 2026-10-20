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
         * Number of Engine status polls (each workerRegistry.health.statusPollIntervalMs apart) allowed
         * after delivery starts before the first matching Engine status must have arrived; the wait is
         * count x poll interval. Heartbeats and delivery acknowledgement do not count. Changes apply to
         * requests registered afterwards.
         */
        private Long dispatchConfirmTimeoutPolls = 4L;
    }

    @Getter
    @Setter
    public static final class DecisionConfig {
        private Double lifetime = 2.0;
    }
}
