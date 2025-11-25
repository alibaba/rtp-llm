package org.flexlb.enums;

import lombok.Getter;

/**
 * 任务生命周期状态枚举
 */
@Getter
public enum TaskStateEnum {
    
    /**
     * 新建状态 - 任务刚添加到本地队列，还未发送给Worker
     */
    CREATED("created"),
    
    /**
     * 在途状态 - 任务已发送给Worker，但还未收到Worker确认
     */
    IN_TRANSIT("in_transit"),

    /**
     * 已确认状态 - Worker已确认收到任务（出现在runningTaskInfo或finishedTaskList中）
     */
    CONFIRMED("confirmed"),
    
    /**
     * 执行中状态 - 任务在Worker的runningTaskInfo中
     */
    RUNNING("running"),
    
    /**
     * 丢失状态 - 任务曾被确认，但现在既不在runningTaskInfo也不在finishedTaskList中
     */
    LOST("lost"),
    
    /**
     * 已完成状态 - 任务出现在finishedTaskList中
     */
    FINISHED("finished"),
    
    /**
     * 被清理状态 - 任务因超时或丢失被强制清理
     */
    CLEANED("timeout_cleaned");
    
    private final String value;
    
    TaskStateEnum(String value) {
        this.value = value;
    }

}