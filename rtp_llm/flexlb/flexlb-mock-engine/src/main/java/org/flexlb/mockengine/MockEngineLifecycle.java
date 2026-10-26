package org.flexlb.mockengine;

import java.util.Map;

/** Hosted worker ownership shared by P/D/V; inference state stays role-specific. */
interface MockEngineLifecycle {
    String getEngineName();
    String getHost();
    String getRoleName();
    int getGrpcPort();
    int getRunningCount();
    int getWaitingCount();
    default int removalWaitingCount() { return getWaitingCount(); }
    boolean isStopped();
    boolean isShuttingDown();
    boolean hasInflightWork();
    Map<String, Integer> inflightWorkSnapshot();
    Map<String, Object> admissionSnapshot();
    void closeAdmission();
    void reopenAdmissionAfterFailedRemoval();
    void setStopped(boolean stopped);
    void drainAndShutdown();
    void shutdown();
    Map<String, Object> getSnapshot();
    Map<String, Object> getTopologySnapshot();
    Map<String, Object> effectiveConfiguration();
}
