package org.flexlb.listener;

/**
 * @author zjw
 * description:
 * date: 2025/3/31
 */
public interface OnlineListener {

    void afterStartUp();

    /**
     * 预热任务优先级
     * @return 值越大，优先级越高
     */
    int priority();

}
