package org.flexlb.state.internal.prefill;

import org.flexlb.state.InternalApi;
import org.flexlb.state.spi.EnginePhase;

/**
 * Prefill 侧相位格：<b>ordinal 即偏序格高度</b>，严格单调链（声明顺序不可重排）。
 *
 * <pre>INIT(0) → ROUTED(1) → QUEUED(2) → DISPATCHING(3) → DISPATCHED(4)
 *   → P_RECEIVED(5) → P_WAITING_UNLOADED(6) → P_WAITING_LOADED(7)
 *   → P_RUNNING(8) → PREFILL_DONE(9)</pre>
 *
 * <p>格内无终态值：终局判定归 {@code org.flexlb.state.TerminalState/TerminalOutcome}
 * （{@link #isTerminal()} 恒 false；PREFILL_DONE 是格顶而非吸收终态）。</p>
 */
@InternalApi
public enum PrefillPhase {

    /** 已创建，尚未路由。 */
    INIT,

    /** 已路由到目标 P 端点。 */
    ROUTED,

    /** 已进入调度队列。 */
    QUEUED,

    /** 正在派发（出队→发送的窗口期）。 */
    DISPATCHING,

    /** 已派发给引擎（发送完成，尚无引擎观察）。 */
    DISPATCHED,

    /** 引擎已收到（保守观察位：PENDING 也映射至此）。 */
    P_RECEIVED,

    /** 引擎已收到但 KV 尚未装载。 */
    P_WAITING_UNLOADED,

    /** KV 已装载、等待执行。 */
    P_WAITING_LOADED,

    /** Prefill 迭代执行中。 */
    P_RUNNING,

    /** Prefill 完成（格顶；终局由上层 settle 成 TerminalState）。 */
    PREFILL_DONE;

    /**
     * 格内无终态：终局判定归 TerminalState/TerminalOutcome，本枚举不承载吸收语义。
     */
    boolean isTerminal() {
        return false;
    }

    /**
     * 引擎观察相位 → P 侧格保守映射：
     * RECEIVED→P_RECEIVED、KV_ALLOCATED→P_WAITING_LOADED、RUNNING→P_RUNNING、
     * PENDING→P_RECEIVED（保守最低观察位：引擎无显式中间相位时只能保守倒推）。
     */
    public static PrefillPhase fromEnginePhase(EnginePhase enginePhase) {
        return switch (enginePhase) {
            case PENDING, RECEIVED -> P_RECEIVED;
            case KV_ALLOCATED -> P_WAITING_LOADED;
            case RUNNING -> P_RUNNING;
        };
    }
}
