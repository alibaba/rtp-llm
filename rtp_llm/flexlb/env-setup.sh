#!/bin/bash

# Flexlb 开发环境自动设置脚本
# 该脚本会自动配置并验证开发环境

echo "🐋 Flexlb 开发环境设置"
echo "=================================="

# 检查 SDKMAN 是否已安装
if [[ -f "$HOME/.sdkman/bin/sdkman-init.sh" ]]; then
    echo "✅ 找到 SDKMAN，正在加载..."
    source "$HOME/.sdkman/bin/sdkman-init.sh"
else
    echo "❌ SDKMAN 未找到，请先安装 SDKMAN:"
    echo "curl -s \"https://get.sdkman.io\" | bash"
    echo "然后重新启动终端或运行: source ~/.sdkman/bin/sdkman-init.sh"
    exit 1
fi

# 检查当前目录
if [[ ! -f "pom.xml" ]]; then
    echo "❌ 必须在项目根目录运行此脚本"
    exit 1
fi

# 检查 .sdkmanrc 是否存在并配置 Java
if [[ -f ".sdkmanrc" ]]; then
    echo "📋 .sdkmanrc 内容:"
    cat .sdkmanrc | grep -v '^#'
    echo
    
    # 自动安装和切换 Java 版本
    echo "🔧 自动安装配置 Java 版本..."
    
    # 安装指定的 Java 版本
    java_version=$(cat .sdkmanrc | grep '^java=' | cut -d'=' -f2 | tr -d ' ')
    if [[ -n "$java_version" ]]; then
        echo "💡 正在检查和安装 Java $java_version..."
        
        # 检查版本是否已安装
        if ! sdk list java | grep -q "$java_version" | grep -q "installed"; then
            echo "⬇️  安装 Java $java_version..."
            sdk install java "$java_version"
        else
            echo "✅ Java $java_version 已安装"
        fi
        
        # 自动启用环境
        sdk env install
        echo "✅ Java 环境已设置为: $java_version"
    fi
else
    echo "⚠️  未找到 .sdkmanrc 文件"
fi

# 验证环境
echo
echo "🔍 验证开发环境:"
echo "----------------------------------"
echo "Java 版本: $(java -version 2>&1 | head -1)"
echo "Maven Wrapper: $(./mvnw --version 2>&1 | head -1)"
echo
echo "🎉 开发环境准备就绪!"
echo
echo "📖 使用说明:"
echo "- 当进入目录时，SDKMAN 会自动切换 Java 版本（需要启用 auto-env）"
echo "- 手动切换: sdk env"
echo "- 构建项目: ./mvnw clean package -DskipTests"