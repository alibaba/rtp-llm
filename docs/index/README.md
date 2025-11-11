# RTP-LLM Website - React + Vite Version

这是 RTP-LLM 官网的 React + Vite 版本，从原始的 HTML/CSS/JS 项目迁移而来，使用 Vite 作为构建工具提供更快的开发体验。

## 功能特性

- ✅ 完全响应式设计
- ✅ 中英文语言切换
- ✅ 现代化 React 组件架构
- ✅ 模块化 CSS 样式
- ✅ 交互式动画效果
- ✅ GitHub 链接跳转

## 项目结构

```
docs/react-app/
├── public/
│   └── index.html          # HTML 模板
├── src/
│   ├── components/         # React 组件
│   │   ├── Header.js       # 头部组件
│   │   ├── Hero.js         # 主要展示区域
│   │   ├── Features.js     # 特性展示区域
│   │   └── FeatureCard.js  # 特性卡片组件
│   ├── hooks/
│   │   └── useLanguage.js  # 语言切换 Hook
│   ├── styles/             # 样式文件
│   │   ├── index.css       # 全局样式
│   │   ├── App.css         # 应用主样式
│   │   ├── Header.css      # 头部样式
│   │   ├── Hero.css        # 主要区域样式
│   │   ├── Features.css    # 特性区域样式
│   │   └── FeatureCard.css # 特性卡片样式
│   ├── App.js              # 主应用组件
│   └── index.js            # 应用入口
├── package.json            # 项目依赖
└── README.md              # 项目说明
```

## 安装和运行

1. 进入项目目录：
```bash
cd docs/react-app
```

2. 安装依赖：
```bash
npm install
```

3. 启动开发服务器：
```bash
npm run dev
# 或者
npm start
```

4. 在浏览器中访问 `http://localhost:3000`

## 构建生产版本

```bash
npm run build
```

构建后的文件将在 `dist/` 目录中。

## 预览生产版本

```bash
npm run preview
```

## 主要组件说明

### Header 组件
- 包含 Logo 和导航按钮
- GitHub 链接跳转
- 语言切换功能

### Hero 组件
- 主标题和副标题展示
- 特性标签（Fast、Flexible、Scalable）
- GitHub 查看按钮和文档按钮

### Features 组件
- 特性网格布局
- 响应式设计

### FeatureCard 组件
- 可复用的特性卡片
- 支持普通图片和复杂图标网格两种模式
- 悬停动画效果

### useLanguage Hook
- 管理中英文切换状态
- 提供翻译函数
- Context API 实现全局状态管理

## 技术栈

- React 18.2.0
- Vite 4.4.5 (构建工具)
- CSS3 (模块化)
- Context API (状态管理)
- React Hooks

## 浏览器支持

- Chrome (最新版本)
- Firefox (最新版本)
- Safari (最新版本)
- Edge (最新版本)

## 部署

- 首先运行打包命令：```npm run build```
- 然后将 `docs/index/dist/` 目录下的内容移动到 `docs/` 目录下（应该只有 `index.html` 和 `assets` 目录）
- 提交到 GitHub
