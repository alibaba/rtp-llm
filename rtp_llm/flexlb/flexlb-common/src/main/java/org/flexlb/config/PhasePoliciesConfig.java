package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Selects grouping and delivery for each request phase.
 * Omitted Generation fields inherit the legacy scheduler and dispatcher configuration.
 */
@Getter
@Setter
public final class PhasePoliciesConfig {
    private Policy encoder = encoderDefaults();
    private Policy generation = new Policy();

    private static Policy encoderDefaults() {
        var policy = new Policy();
        var decision = new DecisionPolicyConfig();
        decision.setType(DecisionPolicyConfig.Type.SINGLE);
        policy.setDecision(decision);
        var delivery = new Delivery();
        delivery.setType(DeliveryType.ROUTE);
        policy.setDelivery(delivery);
        return policy;
    }

    @Getter
    @Setter
    public static final class Policy {
        private DecisionPolicyConfig decision;
        private Delivery delivery;
    }

    /**
     * Selects the phase delivery mechanism; omitted configuration returns a route.
     */
    @Getter
    @Setter
    public static final class Delivery {
        private DeliveryType type = DeliveryType.ROUTE;
    }

    public enum DeliveryType {
        ROUTE,
        ENQUEUE_BATCH
    }
}
