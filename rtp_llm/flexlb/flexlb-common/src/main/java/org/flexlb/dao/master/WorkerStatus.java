package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.util.LoggingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong usedKvCacheTokens = new AtomicLong();
    private CacheStatus cacheStatus;
    private AtomicLong runningQueueTime = new AtomicLong();
    private List<TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1);

    private ConcurrentHashMap<Long/*requestId*/, TaskInfo> localTaskMap = new ConcurrentHashMap<>();
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
        taskInfo.updateTaskState(TaskStateEnum.IN_TRANSIT);

        addRunningQueueTime(taskInfo.estimatePrefillTime());
        lastSelectedTime.set(System.nanoTime() / 1000);
        LoggingUtils.debug("Task {} added to local queue with state: {}", requestId, TaskStateEnum.IN_TRANSIT);
    }

    /**
     * 删除本地运行队列
     * @param requestId 请求ID
     */
    public void removeLocalTask(Long requestId) {
        TaskInfo taskInfo = localTaskMap.get(requestId);
        if (taskInfo != null) {
            addRunningQueueTime(-1 * taskInfo.estimatePrefillTime());
            localTaskMap.remove(requestId);
        }
    }

    /**
     * 添加运行队列中的预估执行时间
     * @param len 要添加的任务的预估执行时间
     */
    public void addRunningQueueTime(long len) {
        runningQueueTime.addAndGet(len);
    }

    public void addKvCacheUsed(long len) {
        usedKvCacheTokens.addAndGet(len);
    }

    public void decKvCacheFree(long len) {
        availableKvCacheTokens.addAndGet(-len);
    }

    /**
     * 更新任务状态
     * 检查丢失、更新运行、清理完成任务
     */
    public void updateTaskStates(List<TaskInfo> runningTaskList, List<TaskInfo> finishedTaskList) {
        // 更新完成任务的版本号
        if (CollectionUtils.isNotEmpty(finishedTaskList)) {
            long maxEndTime = finishedTaskList.stream()
                .mapToLong(TaskInfo::getEndTimeMs)
                .max().orElse(-1);
            if (maxEndTime != -1) {
                latestFinishedTaskVersion.accumulateAndGet(maxEndTime, Math::max);
            }
        }
        
        // 遍历本地任务，并更新任务状态
        Iterator<Map.Entry<Long, TaskInfo>> iterator = localTaskMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Long, TaskInfo> entry = iterator.next();
            Long requestId = entry.getKey();
            TaskInfo localTask = entry.getValue();
            
            // 检查是否在运行列表中
            TaskInfo runningTask = runningTaskList != null ? 
                runningTaskList.stream().filter(t -> t.getInterRequestId() == requestId).findFirst().orElse(null) : null;
                
            // 检查是否在完成列表中
            TaskInfo finishedTask = finishedTaskList != null ? 
                finishedTaskList.stream().filter(t -> t.getInterRequestId() == requestId).findFirst().orElse(null) : null;
            
            // 处理完成的任务
            if (finishedTask != null) {
                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    LoggingUtils.debug("Task {} first confirmed by worker", requestId);
                }
                localTask.updateTaskState(TaskStateEnum.FINISHED);
                
                if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
                    long delta = finishedTask.estimatePrefillTime();
                    safeDecrementQueueTime(runningQueueTime, delta);
                }
                LoggingUtils.debug("Task {} finished and removed", requestId);
                // 本地任务删除Task
                iterator.remove();
                continue;
            }
            
            // 处理运行中的任务
            if (runningTask != null) {
                localTask.setLastActiveTimeUs(System.nanoTime() / 1000);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    LoggingUtils.debug("Task {} first confirmed by worker", requestId);
                }
                if (localTask.getTaskState() != TaskStateEnum.RUNNING) {
                    localTask.updateTaskState(TaskStateEnum.RUNNING);
                }
                
                // 更新引擎返回的字段
                localTask.setPrefixLength(runningTask.getPrefixLength());
                localTask.setPrefillTime(runningTask.getPrefillTime());
                localTask.setInputLength(runningTask.getInputLength());
                localTask.setWaitingTime(runningTask.getWaitingTime());
                localTask.setIterateCount(runningTask.getIterateCount());
                localTask.setEndTimeMs(runningTask.getEndTimeMs());
                localTask.setDpRank(runningTask.getDpRank());
                
                continue;
            }
            
            // 如果任务已经被确认，但是在运行列表和完成列表中都没有，则标记为丢失
            if (localTask.getTaskState() == TaskStateEnum.CONFIRMED || localTask.getTaskState() == TaskStateEnum.RUNNING) {
                localTask.updateTaskState(TaskStateEnum.LOST);
                logger.warn("Task {} marked as LOST - not in running or finished list", requestId);
            }
        }
    }

    /**
     * 更新运行队列的总排队时间
     */
    public void updateRunningQueueTime() {
        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            runningQueueTime.getAndSet(0);
            return;
        }
        long rectifiedEstimateRunningTime = 0;
        for (Entry<Long, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // 基于准确的 cache 命中数重算，纠偏本地任务运行排队时间
            rectifiedEstimateRunningTime += taskInfo.estimatePrefillTime();
        }
        if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
            // 这里仅在纠偏时间小于预估时间时才更新，原因是引擎层返回的 running_list 可能包含排队中的任务，这部分任务的 prefixLength=0
            if (runningQueueTime.get() > rectifiedEstimateRunningTime) {
                runningQueueTime.getAndSet(rectifiedEstimateRunningTime);
            }
        }
    }

    public void updateKvCacheTokens(long latestUsedKvCacheTokens, long latestAvailableKvCacheTokens) {

        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
            availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);
            return;
        }

        long inTransitTaskCacheUsed = 0;
        for (Map.Entry<Long, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // 计算在途Task未命中缓存部分占用的Tokens
            if (taskInfo.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                inTransitTaskCacheUsed = inTransitTaskCacheUsed + taskInfo.getInputLength() - taskInfo.getPrefixLength();
            }
        }
        // 纠偏在途Task影响的KvCache Tokens
        latestUsedKvCacheTokens += inTransitTaskCacheUsed;
        latestAvailableKvCacheTokens -= inTransitTaskCacheUsed;

        usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
        availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);

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
