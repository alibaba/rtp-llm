package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

@Data
@Slf4j
public class WorkerStatus {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private String role;
    private String group;
    private String version;
    private String ip;
    private int port;
    private String site;
    private Long availableConcurrency;
    private boolean alive;
    private AtomicLong kvCacheFree = new AtomicLong();
    private AtomicLong kvCacheUsed = new AtomicLong();
    private CacheStatus cacheStatus;
    private AtomicLong runningQueueTime = new AtomicLong();
    private List<TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1);

    private ConcurrentHashMap<Long, TaskInfo> localTaskMap = new ConcurrentHashMap<>();
    private double stepLatencyMs;
    private long iterateCount;
    private long dpSize;
    private long tpSize;

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1); // 上次更新状态的时间
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1); // 上次更新缓存状态的时间
    private AtomicLong lastSelectedTime = new AtomicLong(-1); // 上次被选中的时间
    private Long statusVersion = -1L;

    /**
     * 添加本地运行队列
     * @param requestId 请求ID
     * @param taskInfo 任务信息
     */
    public void putLocalTask(Long requestId, TaskInfo taskInfo) {
        localTaskMap.put(requestId, taskInfo);
        addRunningQueueTime(taskInfo.estimatePrefillTime());
        lastSelectedTime.set(System.currentTimeMillis());
    }

    /**
     * 删除本地运行队列
     * @param requestId 请求ID
     */
    public void removeLocalTask(Long requestId) {
        TaskInfo taskInfo = localTaskMap.get(requestId);
        addRunningQueueTime(-1 * taskInfo.estimatePrefillTime());
        localTaskMap.remove(requestId);
    }

    /**
     * 添加运行队列中的预估执行时间
     * @param len 要添加的任务的预估执行时间
     */
    public void addRunningQueueTime(long len) {
        runningQueueTime.addAndGet(len);
    }

    public void addKvCacheUsed(long len) {
        kvCacheUsed.addAndGet(len);
    }

    public void decKvCacheFree(long len) {
        kvCacheFree.addAndGet(-len);
    }

    public void updateRunningTaskList(List<TaskInfo> runningTaskList) {
        if (runningTaskList == null) {
            return;
        }

        for (TaskInfo taskInfo : runningTaskList) {
            Long requestId = taskInfo.getInterRequestId();
            localTaskMap.computeIfPresent(requestId, (k, existingTask) -> {
                existingTask.setLastActiveTimeMs(System.currentTimeMillis());
                return taskInfo;
            });
        }
    }

    /**
     * 处理已完成任务
     * @param finishedTaskList 已完成的任务列表
     */
    public void clearFinishedTask(List<TaskInfo> finishedTaskList) {
        if (finishedTaskList == null) {
            return;
        }

        // 原子更新版本号
        long maxEndTime = 0;
        for (TaskInfo taskInfo : finishedTaskList) {
            long endTime = taskInfo.getEndTimeMs();
            if (endTime > maxEndTime) {
                maxEndTime = endTime;
            }
        }
        if (maxEndTime > 0) {
            latestFinishedTaskVersion.accumulateAndGet(maxEndTime, Math::max);
        }

        // 在运行队列中删除已完成任务
        for (TaskInfo taskInfo : finishedTaskList) {
            Long requestId = taskInfo.getInterRequestId();
            localTaskMap.computeIfPresent(requestId, (k, existingTask) -> {
                if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
                    long delta = taskInfo.estimatePrefillTime();
                    safeDecrementQueueTime(runningQueueTime, delta);
                }
                logger.info("Removed task {}", requestId);
                return null; // 返回null表示删除条目
            });
        }
    }

    /**
     * 安全地减少运行队列的总排队时间，确保不会变成负数
     *
     * @param runningQueueTime 运行队列的总排队时间
     * @param timeToReduce 要减少的time
     */
    public static void safeDecrementQueueTime(AtomicLong runningQueueTime, long timeToReduce) {
        if (timeToReduce <= 0) {
            logger.warn("Invalid tokens to reduce: {}", timeToReduce);
            return;
        }
        runningQueueTime.accumulateAndGet(timeToReduce, (currentRunningQueueTime, reductionAmount) -> {
            // 确保减少量为正数，然后计算新值，但不能小于0
            long newRunningQueueTime = currentRunningQueueTime - reductionAmount;

            // 如果计算结果为负数，则设置为0，保证token数量不会小于0
            return Math.max(newRunningQueueTime, 0L);
        });
    }

    /**
     * 获取IP:PORT格式的地址
     *
     * @return IP:PORT字符串
     */
    public String getIpPort() {
        if (ip == null) {
            return null;
        }
        return ip + ":" + port;
    }
}
