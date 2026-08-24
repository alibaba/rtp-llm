package org.flexlb.state.spi;

import java.util.Objects;
import org.flexlb.enums.TaskPhase;

/**
 * 引擎执行相位（规范化视图）。
 *
 * <p>与 flexlb-common 既有 {@link TaskPhase}（org.flexlb.enums，四值）一一对应：
 * 复用既有枚举值域，通过 {@link #fromTaskPhase(TaskPhase)} 静态映射进入本 SPI，
 * state 组件对外只暴露本枚举，不把 dao 层类型泄漏进裁决 API。</p>
 */
public enum EnginePhase {

    /** 引擎已收到请求（尚未分配 KV）。 */
    RECEIVED,

    /** KV 已分配（P 侧 = 已装载等待；D 侧 = LOAD 传输期横跨）。 */
    KV_ALLOCATED,

    /** 正在执行（迭代中）。 */
    RUNNING,

    /** 引擎侧排队等待（保守观察位：映射为两侧格的最低已知观察相位）。 */
    PENDING;

    /**
     * 从既有 {@link TaskPhase}（引擎 status 报文反序列化值域）映射为规范化的 {@link EnginePhase}。
     *
     * @throws NullPointerException taskPhase 为 null（无显式相位时上游必须先做保守倒推，不允许静默吐掉）
     */
    public static EnginePhase fromTaskPhase(TaskPhase taskPhase) {
        Objects.requireNonNull(taskPhase, "taskPhase");
        // TaskPhase 与 EnginePhase 值域一一对应（PENDING/RECEIVED/KV_ALLOCATED/RUNNING）。
        // 引用既有枚举保证值域演进时此处编译期或测试期即暴露，而非运行期静默漂移。
        return switch (taskPhase) {
            case PENDING -> PENDING;
            case RECEIVED -> RECEIVED;
            case KV_ALLOCATED -> KV_ALLOCATED;
            case RUNNING -> RUNNING;
        };
    }
}
