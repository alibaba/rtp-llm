package org.flexlb.state.internal.decode;

import org.flexlb.state.InternalApi;
import org.flexlb.state.spi.EnginePhase;

/**
 * Decode 侧相位格：<b>ordinal 即偏序格高度</b>，严格单调链（声明顺序不可重排）。
 *
 * <pre>RESERVED(0) → DISPATCHED(1) → D_LOADING(2) → D_RUNNING(3)</pre>
 *
 * <p>对齐 E10：KV_ALLOCATED 即 D_LOADING——LOAD 传输期横跨整个 KV 迁移窗口，
 * 不再细分"开始加载/加载中/加载完成"（引擎只暴露 KV_ALLOCATED 一个观察位）。</p>
 *
 * <p>格内无终态值：终局判定归 {@code org.flexlb.state.TerminalState/TerminalOutcome}。</p>
 */
@InternalApi
public enum DecodePhase {

    /** 已预留在目标 D 端点（KV 尚未开始迁移）。 */
    RESERVED,

    /** 已派发迁移/迁移指令已发出（尚无引擎观察；保守观察位：PENDING 也映射至此）。 */
    DISPATCHED,

    /** KV 迁移传输中（引擎观察 KV_ALLOCATED 即进入，横跨整个 LOAD 期，E10）。 */
    D_LOADING,

    /** Decode 迭代执行中（格顶；终局由上层 settle 成 TerminalState）。 */
    D_RUNNING;

    /**
     * 格内无终态：终局判定归 TerminalState/TerminalOutcome，本枚举不承载吸收语义。
     */
    boolean isTerminal() {
        return false;
    }

    /**
     * 引擎观察相位 → D 侧格保守映射：
     * RECEIVED→DISPATCHED、KV_ALLOCATED→D_LOADING、RUNNING→D_RUNNING、
     * PENDING→DISPATCHED（保守最低观察位，L18：引擎无显式中间相位时只能倒推）。
     */
    public static DecodePhase fromEnginePhase(EnginePhase enginePhase) {
        return switch (enginePhase) {
            case PENDING, RECEIVED -> DISPATCHED;
            case KV_ALLOCATED -> D_LOADING;
            case RUNNING -> D_RUNNING;
        };
    }
}
