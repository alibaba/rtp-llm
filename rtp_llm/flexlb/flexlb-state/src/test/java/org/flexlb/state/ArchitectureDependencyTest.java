package org.flexlb.state;

import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.lang.ArchRule;
import org.junit.jupiter.api.Test;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes;
import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses;

/**
 * 架构守护（ArchUnit）：
 * <ol>
 *   <li>依赖红线：flexlb-state 的 main 源码不得依赖 org.flexlb.balance.. / org.flexlb.sync..
 *       （state 是被 balance/sync 消费的底层组件，反向依赖即成环）。</li>
 *   <li>internal 可见性契约（M2）：org.flexlb.state.internal.. 包内类型要么 package-private，
 *       要么 public 且<b>必须</b>带 {@link InternalApi}（门面 StateLedger 跨包协作骨架）。</li>
 *   <li>P3 单写者强制：派生计数器（*Counters）类型 package-private 且仅同包 *SideStore 可依赖
 *       （类型 + 调用位置双约束的架构层固化）。</li>
 *   <li>SPI 纯净性：org.flexlb.state.spi 契约不得依赖 internal 实现。</li>
 * </ol>
 */
class ArchitectureDependencyTest {

    /** 只导入本模块 main 代码（排除测试类，避免测试源码反向校验自身）。 */
    private static final JavaClasses MAIN_CLASSES = new ClassFileImporter()
            .withImportOption(new ImportOption.DoNotIncludeTests())
            .importPackages("org.flexlb.state");

    /** 依赖红线：state 不得依赖 balance / sync。 */
    @Test
    void stateMustNotDependOnBalanceOrSync() {
        ArchRule rule = noClasses()
                .that().resideInAPackage("org.flexlb.state..")
                .should().dependOnClassesThat()
                .resideInAnyPackage("org.flexlb.balance..", "org.flexlb.sync..")
                .because("flexlb-state 是底层状态组件，只允许依赖 flexlb-grpc/flexlb-common；"
                        + "反向依赖 balance/sync 会形成组件环（设计依赖红线）");
        rule.check(MAIN_CLASSES);
    }

    /** internal 可见性契约：public 顶层类型必须 @InternalApi（其余必须 package-private；嵌套类型可见性随宿主）。 */
    @Test
    void internalPublicTypesMustBeAnnotatedWithInternalApi() {
        ArchRule rule = classes()
                .that().resideInAPackage("org.flexlb.state.internal..")
                .and().arePublic()
                .and().areTopLevelClasses()
                .should().beAnnotatedWith(InternalApi.class)
                .because("internal 包是实现细节：package-private 不可见，或 public + @InternalApi "
                        + "仅作为门面 StateLedger 跨包协作骨架（M2 可见性契约；嵌套类型可见性随宿主）");
        rule.check(MAIN_CLASSES);
    }

    /** 防御（多模块预留）：@InternalApi 类型不得被 org.flexlb.state.. 之外的包依赖。 */
    @Test
    void internalApiTypesMustNotLeakOutsideStateModule() {
        ArchRule rule = noClasses()
                .that().resideOutsideOfPackage("org.flexlb.state..")
                .should().dependOnClassesThat()
                .areAnnotatedWith(InternalApi.class)
                .because("@InternalApi 类型是模块实现细节而非对外 API；本模块单仓编译时无外部包（空集恒真），"
                        + "多模块/下游引入 ArchUnit 时复用本规则防泄漏")
                .allowEmptyShould(true);
        rule.check(MAIN_CLASSES);
    }

    /** P3 单写者（类型强制）：派生计数器必须 package-private（@InternalApi 也不允许）。 */
    @Test
    void derivedCountersMustBePackagePrivate() {
        ArchRule rule = classes()
                .that().haveSimpleNameEndingWith("Counters")
                .and().resideInAPackage("org.flexlb.state.internal..")
                .should().notBePublic()
                .because("P3 单写者：计数器 mutator 全 package-private——类型不可见即不可调用，"
                        + "条目/门面/其他组件无法绕过 SideStore 直写计数");
        rule.check(MAIN_CLASSES);
    }

    /** P3 单写者（调用位置强制）：计数器仅同包 *SideStore 可依赖。 */
    @Test
    void derivedCountersAreOnlyReachableBySideStores() {
        ArchRule rule = noClasses()
                .that().resideInAPackage("org.flexlb.state.internal..")
                .and().haveSimpleNameNotEndingWith("SideStore")
                .should().dependOnClassesThat()
                .haveSimpleNameEndingWith("Counters")
                .because("P3 单写者：计数器调用点固定在 SideStore 的 CAS 胜者分支/register/"
                        + "settleRemove 等位置，其他类型（含条目自身）不得依赖计数器");
        rule.check(MAIN_CLASSES);
    }

    /** SPI 纯净性：spi 契约包不得依赖 internal 实现。 */
    @Test
    void spiMustNotDependOnInternal() {
        ArchRule rule = noClasses()
                .that().resideInAPackage("org.flexlb.state.spi")
                .should().dependOnClassesThat()
                .resideInAPackage("org.flexlb.state.internal..")
                .because("spi 是接入契约（EngineObservation/StateEndpointRef 等），"
                        + "依赖 internal 实现会污染对外 API 的稳定性");
        rule.check(MAIN_CLASSES);
    }
}
