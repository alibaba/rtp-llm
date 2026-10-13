package org.flexlb.metric;

import com.sun.management.GarbageCollectionNotificationInfo;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import javax.management.ListenerNotFoundException;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.flexlb.constant.MetricConstant.JVM_GC_COLLECTION_COUNT;
import static org.flexlb.constant.MetricConstant.JVM_GC_PAUSE_TOTAL_MS;

/**
 * Records each JVM GC pause so dashboards can calculate counts and mean pause times per window.
 */
@Slf4j
@Component
public class JvmGcMetricsReporter {

    private final FlexMonitor monitor;
    private final List<GarbageCollectorMXBean> collectors;
    private final Map<String, FlexMetricTags> collectorTags = new LinkedHashMap<>();
    private final List<ListenerRegistration> listeners = new ArrayList<>();

    @Autowired
    public JvmGcMetricsReporter(FlexMonitor monitor) {
        this(monitor, ManagementFactory.getGarbageCollectorMXBeans());
    }

    JvmGcMetricsReporter(FlexMonitor monitor, List<GarbageCollectorMXBean> collectors) {
        this.monitor = monitor;
        this.collectors = List.copyOf(collectors);
    }

    @PostConstruct
    public void init() {
        monitor.register(JVM_GC_COLLECTION_COUNT, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        monitor.register(JVM_GC_PAUSE_TOTAL_MS, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
        String pid = Long.toString(ProcessHandle.current().pid());
        for (GarbageCollectorMXBean collector : collectors) {
            if (collector instanceof NotificationEmitter) {
                collectorTags.put(collector.getName(), FlexMetricTags.of(
                        "gc", gcType(collector.getName()), "collector", collector.getName(), "pid", pid));
            } else {
                log.warn("GC collector does not provide notifications: {}", collector.getName());
            }
        }
        reportHeartbeat();
        for (GarbageCollectorMXBean collector : collectors) {
            if (!(collector instanceof NotificationEmitter emitter)) {
                continue;
            }
            NotificationListener listener = (notification, handback) -> {
                if (!GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION
                        .equals(notification.getType())
                        || !(notification.getUserData() instanceof CompositeData data)) {
                    return;
                }
                GarbageCollectionNotificationInfo info = GarbageCollectionNotificationInfo.from(data);
                reportPause(info.getGcName(), info.getGcInfo().getDuration());
            };
            emitter.addNotificationListener(listener, null, null);
            listeners.add(new ListenerRegistration(emitter, listener));
        }
    }

    private static String gcType(String collector) {
        return switch (collector) {
            // G1's young collector also reports mixed collections that reclaim some old regions.
            case "G1 Young Generation" -> "young";
            case "G1 Old Generation" -> "full";
            // These notifications cover concurrent-cycle pauses, not the whole concurrent cycle.
            case "G1 Concurrent GC" -> "concurrent";
            default -> "other";
        };
    }

    void reportPause(String collector, long durationMs) {
        FlexMetricTags tags = collectorTags.get(collector);
        if (tags == null || durationMs < 0L) {
            return;
        }
        monitor.report(JVM_GC_COLLECTION_COUNT, tags, 1.0D);
        monitor.report(JVM_GC_PAUSE_TOTAL_MS, tags, durationMs);
    }

    /**
     * Keeps zero-event collectors visible and preserves one-second counter samples between pauses.
     */
    @Scheduled(fixedRate = 1000L)
    public void reportHeartbeat() {
        for (FlexMetricTags tags : collectorTags.values()) {
            monitor.report(JVM_GC_COLLECTION_COUNT, tags, 0.0D);
            monitor.report(JVM_GC_PAUSE_TOTAL_MS, tags, 0.0D);
        }
    }

    @PreDestroy
    public void close() {
        for (ListenerRegistration registration : listeners) {
            try {
                registration.emitter().removeNotificationListener(registration.listener());
            } catch (ListenerNotFoundException ignored) {
                // The listener was already removed.
            }
        }
        listeners.clear();
    }

    private record ListenerRegistration(NotificationEmitter emitter, NotificationListener listener) { }
}
