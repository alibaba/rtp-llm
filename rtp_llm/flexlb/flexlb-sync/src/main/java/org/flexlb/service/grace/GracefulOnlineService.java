
package org.flexlb.service.grace;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.listener.AppOnlineHooker;
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

    private static final List<AppOnlineHooker> ONLINE_LISTENERS = new CopyOnWriteArrayList<>();
    @Setter
    private Environment environment;

    public static void addOnlineListener(AppOnlineHooker listener) {
        ONLINE_LISTENERS.add(listener);
    }

    public void online() {
        boolean isTestEnv = Arrays.stream(environment.getActiveProfiles())
                .anyMatch(e -> StringUtils.equals(e, "test"));
        if (isTestEnv) {
            log.info("test env, skip online service");
            return;
        }

        // 预热服务 按照优先级从大到小的顺序
        ONLINE_LISTENERS.sort(Comparator.comparingInt(AppOnlineHooker::priority).reversed());
        for (AppOnlineHooker appOnlineHooker : ONLINE_LISTENERS) {
            appOnlineHooker.afterStartUp();
        }
    }

}
