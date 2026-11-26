package org.flexlb;

import org.flexlb.discovery.ServiceDiscoveryProvider;
import org.flexlb.discovery.ServiceDiscoveryType;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;
import org.springframework.boot.SpringApplication;

import java.util.Map;

/**
 * Runs the standard Master with Mock DP expansion on explicitly configured Whale VIPs.
 */
public final class WhaleMockMasterMain {

    private WhaleMockMasterMain() {}

    public static void main(String[] args) throws Exception {
        var dpByVip =
                MockDpServiceDiscoveryProvider.readLayout(System.getenv("MODEL_SERVICE_CONFIG"));
        if (dpByVip.isEmpty()) {
            Application.main(args);
            return;
        }

        OpenTelemetryBootstrap.configureFromEnvironment();
        var application = new SpringApplication(Application.class);
        application.addInitializers(context -> installDiscovery(context.getBeanFactory(), dpByVip));
        application.run(args);
    }

    static void installDiscovery(
            ConfigurableListableBeanFactory beanFactory, Map<String, Integer> dpByVip) {
        beanFactory.addBeanPostProcessor(
                new BeanPostProcessor() {
                    @Override
                    public Object postProcessAfterInitialization(Object bean, String beanName) {
                        if (bean instanceof ServiceDiscoveryProvider provider
                                && provider.getType() == ServiceDiscoveryType.VIPSERVER) {
                            return new MockDpServiceDiscoveryProvider(provider, dpByVip);
                        }
                        return bean;
                    }
                });
    }
}
