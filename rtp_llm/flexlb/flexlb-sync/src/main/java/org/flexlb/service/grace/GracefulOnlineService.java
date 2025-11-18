
package org.flexlb.service.grace;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.listener.OnlineListener;
import org.flexlb.service.grace.strategy.QueryWarmer;
import org.springframework.context.EnvironmentAware;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Slf4j
@Component
public class GracefulOnlineService implements EnvironmentAware {

    private static final List<OnlineListener> ONLINE_LISTENERS = new CopyOnWriteArrayList<>();
    @Setter
    private Environment environment;

    public static void addOnlineListener(OnlineListener listener) {
        ONLINE_LISTENERS.add(listener);
    }

    public void online(String profile) {
        log.info("warm up start");
        boolean isTestEnv = Arrays.stream(environment.getActiveProfiles())
                .anyMatch(e -> StringUtils.equals(e, "test"));
        if (isTestEnv) {
            log.info("warm up in test env, skip");
            return;
        }
        if (StringUtils.equals(profile, "test")) {
            log.info("no need warm up");
            QueryWarmer.warmUpFinished = true;
            return;
        }

        // 预热服务 按照优先级从大到小的顺序
        ONLINE_LISTENERS.sort(Comparator.comparingInt(OnlineListener::priority).reversed());
        for (OnlineListener onlineListener : ONLINE_LISTENERS) {
            onlineListener.afterStartUp();
        }
    }

}
