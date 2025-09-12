package org.flexlb;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.grace.GracefulOnlineService;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.Environment;

/**
 * Pandora Boot应用的入口类
 */
@Slf4j
@SpringBootApplication(scanBasePackages = {"org.flexlb"},
        exclude = RedisAutoConfiguration.class
)
public class Application {

    public static void main(String[] args) {
        // 打印启动参数
        log.info("Application start with args: {}", (Object[]) args);
        ConfigurableApplicationContext context = SpringApplication.run(Application.class, args);
        Environment env = context.getEnvironment();
        String port = env.getProperty("server.port");
        String profile = env.getProperty("spring.profiles.active");
        log.info("flex-lb server started on port {}", port);

        // Get GracefulOnlineService instance from context and call online
        GracefulOnlineService gracefulOnlineService = context.getBean(GracefulOnlineService.class);
        gracefulOnlineService.online(profile);
    }
}