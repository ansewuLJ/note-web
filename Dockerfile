# 使用官方Python镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /docs

# 配置清华源镜像加速
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list

# 配置pip清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装MkDocs和Material主题以及必要的插件
RUN pip install --no-cache-dir \
    mkdocs \
    mkdocs-material \
    pymdown-extensions \
    mkdocs-awesome-pages-plugin \
    mkdocs-minify-plugin

# 复制文档文件
COPY . /docs

# 暴露端口
EXPOSE 8000

# 启动MkDocs开发服务器
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]
