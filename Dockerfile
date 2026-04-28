# =============================================
# Stage 1: Build frontend
# =============================================
FROM node:22-alpine AS frontend-builder

WORKDIR /build/frontend

# 安装所有依赖（包含 devDependencies，因为构建需要 vite）
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci && npm cache clean --force

# 复制前端源码并构建
COPY frontend/ ./
RUN npm run build

# =============================================
# Stage 2: Build backend
# =============================================
FROM python:3.11-slim AS backend-builder

WORKDIR /build/backend

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制依赖文件并安装
COPY backend/pyproject.toml .
RUN uv sync --no-dev

# =============================================
# Stage 3: Final runtime image
# =============================================
FROM python:3.11-slim

WORKDIR /app

# 安装 uv
RUN pip install --no-cache-dir uv

# 从 backend-builder 复制虚拟环境和源码
COPY --from=backend-builder /build/backend/.venv /app/backend/.venv
COPY --from=backend-builder /build/backend/pyproject.toml /app/backend/

# 复制后端源码
COPY backend/ ./backend/

# 复制前端构建产物
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# 复制项目配置和静态文件
COPY package.json package-lock.json ./
COPY frontend/package.json ./frontend/
COPY static/ ./static/

EXPOSE 5001

# 启动后端（前端静态文件由后端服务）
CMD ["/app/backend/.venv/bin/python", "backend/run.py"]
