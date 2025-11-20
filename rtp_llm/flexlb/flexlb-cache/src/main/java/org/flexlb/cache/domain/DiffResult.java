package org.flexlb.cache.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serial;
import java.io.Serializable;
import java.util.Collections;
import java.util.Set;

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

    @Serial
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