package org.flexlb.listener;

public interface AppOnlineHooker {

    /**
     * Application startup callback
     */
    void afterStartUp();

    /**
     * Warmup task priority
     * @return Higher value means higher priority
     */
    int priority();

}
