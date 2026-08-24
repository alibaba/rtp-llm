package org.flexlb.state;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 模块内部公开骨架标记：标注 org.flexlb.state.internal.. 包内需要被门面
 * {@code StateLedger}（org.flexlb.state 包）跨包协作的类型。
 *
 * <p>可见性约定（ArchUnit 守护）：</p>
 * <ul>
 *   <li>internal 包内类型：要么 package-private，要么 public 且<b>必须</b>带本注解；</li>
 *   <li>带本注解的类型不得被 {@code org.flexlb.state..} 之外的任何包依赖
 *       （它们是模块实现细节，不是对外 API）。</li>
 * </ul>
 */
@Documented
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface InternalApi {
}
