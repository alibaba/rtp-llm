package org.flexlb.sync;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 清理异步请求的Disposable，避免内存泄漏问题
 */
@Component
public class StatusQueryDisposableCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    public static AtomicLong updateTime = new AtomicLong(0);

    public static long CLEAN_DISPOSABLE_PERIOD = 3 * 1000;

    public StatusQueryDisposableCleaner() {
        ScheduledThreadPoolExecutor DisposableCleanScheduler = new ScheduledThreadPoolExecutor(5,
                new NamedThreadFactory("disposable-clean-scheduler"), new ThreadPoolExecutor.AbortPolicy());
        DisposableCleanScheduler.scheduleAtFixedRate(() -> {
            logger.info("scheduled clean queryStatusDisposableMap, size: {}", EngineWorkerStatus.queryStatusDisposableMap.size());
            cleanDisposable();
        }, 0, CLEAN_DISPOSABLE_PERIOD, TimeUnit.MILLISECONDS);
    }

    /**
     * 清理超过3秒的Disposable对象
     */
    public static void cleanDisposable() {
        Map<String, Disposable> queryStatusDisposableMap = EngineWorkerStatus.queryStatusDisposableMap;
        // 遍历查询状态Disposable映射表中的所有条目
        for (Iterator<Map.Entry<String, Disposable>> it = queryStatusDisposableMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, Disposable> entry = it.next();
            String key = entry.getKey();
            String[] splitKey = key.split("@");
            String timestampStr = splitKey[0];

            Disposable disposable = entry.getValue();

            long currentTime = System.currentTimeMillis();

            // 如果当前时间减去时间戳超过指定时间，则清理该Disposable对象
            if ((currentTime - Long.parseLong(timestampStr)) > CLEAN_DISPOSABLE_PERIOD) {
                if (disposable != null) {
                    // 释放资源
                    disposable.dispose();
                }

                it.remove();
            }
        }
        logger.info("queryStatusDisposableMap after size: {}", queryStatusDisposableMap.size());
    }

}
