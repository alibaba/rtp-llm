package org.flexlb.util.discovery;

import java.util.List;

/**
 * ServiceHostListener - Service host change listener
 * Callback triggered when service host list changes
 *
 * @author saichen.sm
 */
@FunctionalInterface
public interface ServiceHostListener {

    /**
     * Triggered when host list changes
     *
     * @param hosts New host list
     */
    void onHostsChanged(List<DiscoveredHost> hosts);
}
