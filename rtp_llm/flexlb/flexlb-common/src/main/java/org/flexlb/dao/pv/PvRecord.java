package org.flexlb.dao.pv;

import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;

/**
 * Common emission contract for the PV record types sharing {@code pv.log}
 * ({@link PvLogData}, {@link BatchPvLogData}, {@link DispatchPvLogData}). Success records
 * use INFO, non-success uses ERROR. Serialization failure surfaces to the operational log
 * (not pv.log) so ops dashboards see it; PV emission must never propagate an exception
 * into the request path.
 */
public interface PvRecord {

    /**
     * The one appender all PV records share. Keeping the logger name here (instead of each
     * emitting class re-declaring {@code getLogger("pvLogger")}) makes it impossible to point
     * a PV record at the wrong appender.
     */
    org.slf4j.Logger PV_LOGGER = org.slf4j.LoggerFactory.getLogger("pvLogger");

    boolean isSuccess();

    /** Emit to the shared {@code pv.log} appender. */
    default void emit() {
        String json;
        try {
            json = JsonUtils.toString(this);
        } catch (Exception ex) {
            Logger.error("Failed to serialize PV log data", ex);
            return;
        }
        if (isSuccess()) {
            PV_LOGGER.info(json);
        } else {
            PV_LOGGER.error(json);
        }
    }
}
