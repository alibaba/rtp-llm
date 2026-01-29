package org.flexlb.listener;

public interface AppOnlineHooker {

    /**
     * 应用启动回调
     */
    void afterStartUp();

    /**
     * 预热任务优先级
     * @return 值越大，优先级越高
     */
    int priority();

}
