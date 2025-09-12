package org.flexlb.dao.master;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private AtomicLong expirationTime = new AtomicLong(-1);
    private AtomicLong lastUpdateTime = new AtomicLong(-1);
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1);
    private AtomicLong lastScheduleTime = new AtomicLong(-1);
    private Long statusVersion = -1L;
    private static long TASK_TIME_OUT_MS = 1000 * 60;

    public void putLocalTask(Long requestId, TaskInfo taskInfo) {
        localTaskMap.put(requestId, taskInfo);
        addRunningQueueTime(taskInfo.estimatePrefillTime());
        lastScheduleTime.set(System.currentTimeMillis());
    }

    public void removeLocalTask(Long requestId) {
        TaskInfo taskInfo = localTaskMap.get(requestId);
        addRunningQueueTime(-1 * taskInfo.estimatePrefillTime());
        localTaskMap.remove(requestId);
    }

    public void addRunningQueueTime(long len) {
        runningQueueTime.addAndGet(len);
    }

    /**
     * 安全地减少队列中的token数量，确保不会变成负数
     * 
     * @param timeToReduce 要减少的time
     */
    private void safeDecrementQueueTime(long timeToReduce) {
        if (timeToReduce <= 0) {
            logger.warn("Invalid tokens to reduce: {}", timeToReduce);
            return;
        }
        runningQueueTime.accumulateAndGet(timeToReduce, (currentTokens, reductionAmount) -> {
            // 确保减少量为正数，然后计算新值，但不能小于0
            long newTokenCount = currentTokens - reductionAmount;
            
            // 如果计算结果为负数，则设置为0，保证token数量不会小于0
            return Math.max(newTokenCount, 0L);
        });
    }

    public void addKvCacheUsed(long len) {
        kvCacheUsed.addAndGet(len);
    }

    public void decKvCacheFree(long len) {
        kvCacheFree.addAndGet(-len);
    }

    public void clearFinishedTaskAndTimeoutTask(List<TaskInfo> finishedTaskList) {
        handelFinishedTask(finishedTaskList);
        handleTimeoutTasks(System.currentTimeMillis());
    }

    private void handelFinishedTask(List<TaskInfo> finishedTaskList) {
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

        // 原子处理每个任务
        for (TaskInfo taskInfo : finishedTaskList) {
            Long requestId = taskInfo.getInterRequestId();
            localTaskMap.computeIfPresent(requestId, (k, existingTask) -> {
                if (RoleType.PREFILL.matches(role)) {
                    long delta = taskInfo.estimatePrefillTime();
                    safeDecrementQueueTime(delta);
                }
                logger.info("Removed task {}", requestId);
                return null;
            });
        }
    }
    private void handleTimeoutTasks(long currentTime) {
        // 原子遍历检测超时
        localTaskMap.forEach((requestId, task) -> {
            if (isTaskTimeout(task, currentTime)) {
                localTaskMap.computeIfPresent(requestId, (k, existing) -> {
                    logger.warn("Removing timeout task: {}", requestId);
                    return removeAndUpdateQueueTime(existing);
                });
            }
        });
    }

    private TaskInfo removeAndUpdateQueueTime(TaskInfo task) {
        if (RoleType.PREFILL.matches(role)) {
            long delta = task.estimatePrefillTime();
            safeDecrementQueueTime(delta);
        }
        return null; // 返回null表示删除条目
    }

    private boolean isTaskTimeout(TaskInfo task, long currentTime) {
        // 使用任务开始时间或创建时间判断超时
        long taskStartTime = task.getEnqueueTimeMs();
        return (currentTime - taskStartTime) > TASK_TIME_OUT_MS;
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
