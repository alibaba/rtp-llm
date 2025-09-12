package org.flexlb.balance;

import org.flexlb.balance.scheduler.Scheduler;
import org.flexlb.enums.ScheduleType;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class SchedulerFactory {

    private final static Map<ScheduleType, Scheduler> prefillLoadBalancerFactory = new ConcurrentHashMap<>();

    public static void register(ScheduleType strategy, Scheduler prefillScheduler) {
        prefillLoadBalancerFactory.put(strategy, prefillScheduler);
    }

    public static Scheduler getScheduler(ScheduleType strategy) {
        Scheduler prefillLoadBalancer = prefillLoadBalancerFactory.get(strategy);
        if (prefillLoadBalancer == null) {
            throw new RuntimeException("scheduleType not found: " + strategy);
        }
        return prefillLoadBalancer;
    }
}
