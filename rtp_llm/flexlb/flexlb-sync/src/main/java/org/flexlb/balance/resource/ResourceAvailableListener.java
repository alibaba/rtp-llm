package org.flexlb.balance.resource;

/**
 * 资源可用监听器
 *
 * @author saichen.sm
 * @since 2025/12/24
 */
public interface ResourceAvailableListener {

    /**
     * 当资源从不可用变为可用时触发
     */
    void onResourceAvailable();

}
