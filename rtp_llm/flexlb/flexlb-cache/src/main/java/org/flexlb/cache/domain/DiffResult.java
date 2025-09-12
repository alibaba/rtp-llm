package org.flexlb.cache.domain;

import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Diff计算结果
 * 描述两个缓存状态之间的差异
 * 
 * @author FlexLB
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DiffResult implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 新增的缓存块
     */
    @Builder.Default
    private Set<Long> addedBlocks = Collections.emptySet();
    
    /**
     * 删除的缓存块
     */
    @Builder.Default
    private Set<Long> removedBlocks = Collections.emptySet();
    
    
    /**
     * 引擎IP
     */
    private String engineIp;
    
    /**
     * 计算时间戳
     */
    @Builder.Default
    private long calculateTime = System.currentTimeMillis();
    
    /**
     * 版本号
     */
    private String version;
    
    /**
     * 检查是否有变化
     */
    public boolean hasChanges() {
        return !addedBlocks.isEmpty() || !removedBlocks.isEmpty();
    }
    
    /**
     * 创建空的Diff结果
     */
    public static DiffResult empty(String engineIp) {
        return DiffResult.builder()
                .engineIp(engineIp)
                .addedBlocks(Collections.emptySet())
                .removedBlocks(Collections.emptySet())
                .build();
    }
}