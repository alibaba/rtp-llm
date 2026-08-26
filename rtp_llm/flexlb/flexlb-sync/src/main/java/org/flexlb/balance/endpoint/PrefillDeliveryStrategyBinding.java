package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryStrategy;

/** Startup-selected delivery behavior shared by every Prefill generation. */
public interface PrefillDeliveryStrategyBinding {

    /** Return the single immutable strategy selected during composition. */
    DeliveryStrategy strategy();
}
