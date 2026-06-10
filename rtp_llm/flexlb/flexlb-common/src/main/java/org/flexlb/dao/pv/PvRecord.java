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

    boolean isSuccess();

    default void emit(org.slf4j.Logger pvLogger) {
        try {
            String json = JsonUtils.toStringOrEmpty(this);
            if (isSuccess()) {
                pvLogger.info(json);
            } else {
                pvLogger.error(json);
            }
        } catch (Exception ex) {
            Logger.error("Failed to serialize PV log data", ex);
        }
    }
}
