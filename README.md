# 📚 我的笔记本

使用 MkDocs Material 构建的个人笔记系统，支持数学公式和图片显示。

## 🚀 快速启动

**首先进入项目目录：**
```bash
cd note-web
```

### 方法一：Docker Compose（推荐）

```bash
# 启动服务
docker-compose up -d

# 访问 http://localhost:8000
```

### 方法二：直接 Docker

```bash
# 构建镜像
docker build -t my-notes .

# 运行容器
docker run -d -p 8000:8000 -v ${PWD}:/docs --name my-notes my-notes

# 访问 http://localhost:8000
```

## ✨ 功能特性

- 🎨 **美观界面**：Material Design 风格
- 📐 **数学公式**：支持 LaTeX 语法 `$...$` 和 `$$...$$`
- 🖼️ **图片显示**：自动优化和响应式
- 🔍 **全文搜索**：中文搜索支持
- 🌙 **深色模式**：自动切换
- 📱 **响应式**：手机端友好
- 🚀 **快速加载**：资源压缩优化

## 📝 使用说明

1. 将 Markdown 文件放在 `docs/` 目录下
2. 图片放在相应的子目录中
3. 数学公式使用 `$公式$` 或 `$$公式$$` 格式
4. 修改文件后自动刷新页面

## 🛠️ 管理命令

**注意：所有命令都需要在项目目录下执行**

```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建
docker-compose up --build

# 查看日志
docker-compose logs -f
```

现在访问 **http://localhost:8000** 即可查看您的笔记！
