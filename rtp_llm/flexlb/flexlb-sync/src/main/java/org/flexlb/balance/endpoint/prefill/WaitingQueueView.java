package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * 等待队列的一份不可变拷贝。持锁时生成，里面的 items 和 plannerItems 在锁外第一次用到时才构建。
 */
public final class WaitingQueueView {
    public static final WaitingQueueView EMPTY = new WaitingQueueView(List.of());
    private final List<PrefillActiveIndex.Entry> entries;
    private volatile List<ScheduledRequest> items;
    private volatile List<GroupPlanner.Item> plannerItems;

    public WaitingQueueView(Collection<PrefillActiveIndex.Entry> entries) {
        this.entries = List.copyOf(entries);
    }

    public List<ScheduledRequest> items() {
        List<ScheduledRequest> result = items;
        if (result != null) {
            return result;
        }
        synchronized (this) {
            if (items == null) {
                List<ScheduledRequest> requests = new ArrayList<>(entries.size());
                for (PrefillActiveIndex.Entry entry : entries) {
                    requests.add(entry.request);
                }
                items = List.copyOf(requests);
            }
            return items;
        }
    }

    public List<GroupPlanner.Item> plannerItems() {
        List<GroupPlanner.Item> result = plannerItems;
        if (result != null) {
            return result;
        }
        synchronized (this) {
            if (plannerItems == null) {
                List<GroupPlanner.Item> resultItems = new ArrayList<>(entries.size());
                for (PrefillActiveIndex.Entry entry : entries) {
                    resultItems.add(entry.plannerItem());
                }
                plannerItems = List.copyOf(resultItems);
            }
            return plannerItems;
        }
    }
}
