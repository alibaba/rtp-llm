package org.flexlb.balance.resource;

import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * 资源度量器工厂
 * 根据RoleType获取对应的资源度量器
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class ResourceMeasureFactory {

    private final Map<ResourceMeasureIndicatorEnum, ResourceMeasure> measureMap;

    public ResourceMeasureFactory(List<ResourceMeasure> measureList) {
        this.measureMap = new EnumMap<>(ResourceMeasureIndicatorEnum.class);
        for (ResourceMeasure measure : measureList) {
            measureMap.put(measure.getResourceMeasureIndicator(), measure);
        }
    }

    /**
     * 根据角色类型获取对应的资源度量器
     *
     * @param measureIndicator 资源度量指标
     * @return 资源度量器,如果没有则返回null
     */
    public ResourceMeasure getMeasure(ResourceMeasureIndicatorEnum measureIndicator) {
        return measureMap.get(measureIndicator);
    }
}
