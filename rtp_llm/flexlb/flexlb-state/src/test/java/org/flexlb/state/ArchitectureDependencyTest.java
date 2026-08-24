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
 *   <li>包可见性：org.flexlb.state.internal 包内类型必须 package-private（不对外泄漏实现）。</li>
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

    /** 包可见性：internal 实现类型必须 package-private。 */
    @Test
    void internalTypesMustNotBePublic() {
        ArchRule rule = classes()
                .that().resideInAPackage("org.flexlb.state.internal..")
                .should().notBePublic()
                .because("internal 包是实现细节（相位格纯函数、轨迹环），"
                        + "对外只暴露 org.flexlb.state / org.flexlb.state.spi 契约");
        rule.check(MAIN_CLASSES);
    }
}
