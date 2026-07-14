# Paralegal Agent - 智能律师助理 AI

> 基于 CrewAI 多智能体协作的法律文档分析系统，采用两阶段检索（二进制粗过滤 + 精确重排序）与 Milvus 向量数据库 RAG 架构，实现带引用溯源的智能法律问答。

**GitHub 仓库**: https://github.com/zike678/paralegal-agent

---

## 1. 架构说明

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层 (Streamlit)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    app.py (主入口)                       │   │
│  │  - PDF 上传与预览  - 聊天交互  - 引用展示  - 日志面板    │   │
│  └──────────────────────────┬──────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     工作流编排层 (CrewAI Flows)                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          ParalegalAgentWorkflow (agent_workflow.py)      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │   │
│  │  │  检索     │→│ RAG 生成  │→│ 质量评估  │→│ 综合回答 │  │   │
│  │  │ Retrieve │ │ Generate │ │ Evaluate │ │Synthesi│  │   │
│  │  └──────────┘  └──────────┘  └─────┬────┘  └────────┘  │   │
│  │                                    │                     │   │
│  │                              质量不达标？                  │   │
│  │                                    │ YES                 │   │
│  │                                    ▼                     │   │
│  │                              ┌──────────┐                │   │
│  │                              │ 网页搜索  │                │   │
│  │                              │Web Search│                │   │
│  │                              └─────┬────┘                │   │
│  │                                    │                     │   │
│  │                                    └──────→ 综合回答      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
     ┌─────────┐         ┌─────────┐         ┌─────────┐
     │ 检索层   │         │ 生成层   │         │ 工具层   │
     │retrieval│         │generation│         │  tools  │
     └────┬────┘         └────┬────┘         └────┬────┘
          │                   │                   │
          ▼                   ▼                   ▼
     ┌─────────┐         ┌─────────┐         ┌─────────┐
     │ Milvus  │         │  LLM    │         │Firecrawl│
     │向量数据库│         │ 模型    │         │网页搜索  │
     │(二进制量化)│        │         │         │         │
     └─────────┘         └─────────┘         └─────────┘
          ▲
          │
     ┌─────────┐
     │ 向量化层 │
     │embedding│
     └────┬────┘
          │
     ┌─────────┐
     │ 索引层   │
     │indexing │
     └─────────┘
```

### 1.2 核心模块说明

| 模块 | 文件路径 | 职责 |
|------|---------|------|
| **主入口** | `app.py` | Streamlit 应用主界面，PDF 上传、聊天交互、引用展示 |
| **工作流** | `src/workflow/agent_workflow.py` | CrewAI Flow 编排，检索→生成→评估→搜索→综合 |
| **事件定义** | `src/workflow/event.py` | 工作流各阶段事件类型定义 |
| **检索模块** | `src/retrieval/retriever_rerank.py` | 两阶段检索：二进制粗过滤 + 精确重排序 |
| **生成模块** | `src/generation/rag.py` | RAG 问答生成 |
| **向量化模块** | `src/embeddings/embed_data.py` | 文本嵌入生成，支持二进制量化 |
| **索引模块** | `src/indexing/milvus_vdb.py` | Milvus 向量数据库操作封装 |
| **工具模块** | `src/tools/firecrawl_search_tool.py` | Firecrawl 网页搜索工具 |
| **配置管理** | `config/settings.py` | 应用配置集中管理 |

### 1.3 技术栈

- **前端框架**: Streamlit (快速构建数据应用)
- **Agent 框架**: CrewAI + CrewAI Flows (工作流编排)
- **向量数据库**: Milvus (高性能向量检索，支持二进制量化)
- **网页搜索**: Firecrawl (网页内容爬取与解析)
- **LLM**: OpenAI API / Ollama (兼容 OpenAI 接口的本地模型)
- **PDF 处理**: PyPDF
- **依赖管理**: uv (Python 包管理)
- **日志**: loguru

---

## 2. 关键 Prompt 与 Vibe 思路

### 2.1 路由评估提示词 (ROUTER_EVALUATION_TEMPLATE)

**核心设计思路**:
- **质量守门员**: 作为 RAG 回答的质量把关者，判断回答是否足够好
- **二元判定**: 简化为 "GOOD" / "BAD" 二元输出，便于程序路由
- **多维度评估**: 从相关性、事实一致性、详细程度、诚实性四个维度评估
- **严格标准**: 宁可触发网页搜索补充，也不输出低质量回答

**Vibe 关键词**: `严格把关` `质量优先` `宁缺毋滥` `客观评判`

### 2.2 查询优化提示词 (QUERY_OPTIMIZATION_TEMPLATE)

**核心设计思路**:
- **搜索思维**: 将用户自然语言问题转化为搜索引擎友好的查询词
- **关键词增强**: 添加法律领域专业术语，提高搜索相关性
- **权威性导向**: 优化后的查询更可能找到官方、权威的法律来源
- **简洁全面**: 在精简和全面之间找到平衡

**Vibe 关键词**: `搜索优化` `专业术语` `权威导向` `精准命中`

### 2.3 综合回答提示词 (SYNTHESIS_TEMPLATE)

**核心设计思路**:
- **信息融合**: 智能融合文档知识库和网页搜索结果
- **来源标注**: 明确区分信息来自文档还是网络，保持透明度
- **矛盾处理**: 当不同来源信息冲突时，诚实指出并说明
- **优先级**: 优先采用可靠来源的信息

**Vibe 关键词**: `融合创新` `来源透明` `客观中立` `专业严谨`

### 2.4 RAG 系统提示词

**核心设计思路**:
- **法律专业性**: 使用法律专业术语，保持严谨性
- **引用溯源**: 每个结论都标注来源段落，支持查证
- **谨慎表述**: 使用"根据文档内容"、"可能"等谨慎措辞
- **结构化输出**: 法律分析通常采用"结论-依据-分析"结构

**Vibe 关键词**: `专业严谨` `有据可依` `条理清晰` `客观中立`

### 2.5 Prompt 设计总原则

1. **法律严谨性**: 所有提示词都强调法律领域的专业性和严谨性
2. **可追溯性**: 强调引用来源，每个结论都要有依据
3. **诚实原则**: 不知道就说不知道，不编造法律条文或案例
4. **风险意识**: 法律问题涉及风险，提示词中体现谨慎态度

---

## 3. AI 调用逻辑

### 3.1 工作流执行流程

```
用户提问
    │
    ▼
┌─────────────────────────────────────────┐
│  1. 检索阶段 (Retrieve)                  │
│  - 二进制量化粗过滤（快速召回）          │
│  - 精确向量重排序（精排 Top-K）          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  2. RAG 生成阶段 (Generate)              │
│  - 基于检索上下文生成初步回答            │
│  - 引用相关文档片段                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  3. 质量评估阶段 (Evaluate)              │
│  - LLM 评估回答质量                      │
│  - 输出 GOOD / BAD                       │
└──────────┬───────────┬──────────────────┘
           │           │
        GOOD         BAD
           │           │
           ▼           ▼
       直接输出    ┌──────────────────────┐
                   │  4. 网页搜索阶段       │
                   │  - 查询词优化          │
                   │  - Firecrawl 搜索      │
                   └───────────┬──────────┘
                               │
                               ▼
                   ┌──────────────────────┐
                   │  5. 综合回答阶段       │
                   │  - 融合文档+网页信息   │
                   │  - 生成最终回答        │
                   └───────────┬──────────┘
                               │
                               ▼
                           最终输出
```

### 3.2 两阶段检索机制

**第一阶段: 二进制粗过滤 (Binary Coarse Filtering)**

```python
# 原理: 将向量二值化为 0/1，使用汉明距离快速计算
# 优势: 速度快 10-100 倍，内存占用小 32 倍
# 适用: 百万级向量快速召回候选集

def binary_search(query_vector, top_k=100):
    # 二值化查询向量
    binary_query = binarize(query_vector)
    # 汉明距离快速排序
    candidates = hamming_distance_sort(binary_query, top_k=100)
    return candidates
```

**第二阶段: 精确重排序 (Precise Reranking)**

```python
# 原理: 对粗过滤候选集使用原始浮点向量计算余弦相似度
# 优势: 保持高精度
# 适用: 小范围精排，保证最终结果质量

def precise_rerank(query_vector, candidates, top_k=3):
    # 使用原始浮点向量计算精确相似度
    scores = cosine_similarity(query_vector, candidates)
    # 重排序，返回 Top-K
    return sort_by_score(scores, top_k=3)
```

**性能收益**:
- 召回速度提升: 10-100x
- 内存占用降低: 32x (float32 → binary)
- 精度损失: < 5%（通过重排序弥补）

### 3.3 Function Calling / 工具调用

**已实现工具**:

| 工具名称 | 功能说明 | 调用方式 |
|---------|---------|---------|
| `FirecrawlSearchTool` | 网页搜索与内容爬取 | Firecrawl API |
| `Retriever.search` | 向量数据库检索 | Milvus 相似度查询 |
| `RAG.query` | RAG 问答生成 | LLM Completion |

**工具调用机制**:
- 工作流驱动: 由 CrewAI Flow 状态机控制调用时机
- 条件触发: 网页搜索仅在 RAG 回答质量不足时触发
- 结果融合: 多源信息通过 LLM 综合后输出

### 3.4 流式输出支持

**当前实现**:
- Streamlit 界面支持 `st.spinner` 加载状态
- 工作流日志实时捕获与展示
- 引用来源可展开查看

**扩展能力**:
- 可通过 SSE 实现真正的流式输出
- 支持打字机效果展示回答
- 可逐步展示检索、生成、搜索各阶段状态

### 3.5 引用溯源机制

**实现方式**:
1. 检索阶段记录每个文档块的 ID 和内容
2. 生成阶段在上下文中包含来源信息
3. 展示阶段提取引用，可展开查看原文片段
4. 显示相似度分数，帮助用户判断可信度

**引用格式**:
```
[1] score=0.892 id=doc_001
  "文档片段内容..."
```

---

## 4. 部署步骤说明

### 4.1 环境要求

- Python 3.10+
- 内存: 建议 8GB 以上（Milvus + 向量计算）
- 磁盘: 建议 20GB 以上（向量数据库 + PDF 存储）
- 可选: GPU（加速向量化，非必需）

### 4.2 本地部署

#### 步骤 1: 克隆仓库

```bash
git clone https://github.com/zike678/paralegal-agent.git
cd paralegal-agent
```

#### 步骤 2: 安装依赖（使用 uv）

```bash
# 安装 uv（如果未安装）
pip install uv

# 使用 uv 安装依赖
uv sync
```

或者使用 pip:

```bash
pip install -r requirements.txt
```

#### 步骤 3: 配置环境变量

```bash
# 创建 .env 文件
cat > .env << EOF
# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key

# Firecrawl API 配置（可选，用于网页搜索）
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Ollama 配置（可选，使用本地模型）
# OLLAMA_BASE_URL=http://localhost:11434
# LLM_MODEL=gpt-oss:20b

# Milvus 配置
MILVUS_DB_PATH=./milvus_db
COLLECTION_NAME=paralegal_docs

# 向量配置
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
VECTOR_DIM=768
TOP_K=3

# 生成配置
TEMPERATURE=0.1
MAX_TOKENS=2048
EOF
```

#### 步骤 4: 启动应用

```bash
# 激活虚拟环境（如果使用 uv）
source .venv/bin/activate

# 启动 Streamlit
streamlit run app.py
```

应用启动后访问: `http://localhost:8501`

#### 步骤 5: 使用流程

1. 在侧边栏配置 API Key
2. 上传 PDF 法律文档
3. 等待文档处理完成（向量化 + 建索引）
4. 在聊天框提问，获取带引用的回答

### 4.3 生产环境部署（含 DNS/HTTPS）

#### 方案一: Docker 部署

##### 4.3.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv

# 复制项目文件
COPY . .

# 安装依赖
RUN uv sync --frozen

# 暴露端口
EXPOSE 8501

# 健康检查
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 启动命令
ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

##### 4.3.2 docker-compose.yml（推荐）

```yaml
version: '3.8'

services:
  paralegal-agent:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - MILVUS_DB_PATH=/app/data/milvus_db
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

##### 4.3.3 启动服务

```bash
# 创建环境变量文件
cp .env.example .env
# 编辑 .env 填入真实密钥

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

#### 4.3.4 DNS 配置

1. 在域名服务商添加 A 记录:
   - `paralegal.yourdomain.com` → 服务器公网 IP

2. 验证 DNS 解析:
   ```bash
   ping paralegal.yourdomain.com
   ```

##### 4.3.5 Nginx 反向代理配置

```nginx
server {
    listen 80;
    server_name paralegal.yourdomain.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name paralegal.yourdomain.com;

    # SSL 证书
    ssl_certificate /etc/letsencrypt/live/paralegal.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/paralegal.yourdomain.com/privkey.pem;

    # SSL 安全配置
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;

    # 文件上传大小限制
    client_max_body_size 100M;

    # Streamlit 特定配置
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 支持
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        proxy_buffering off;
    }

    # 静态资源缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        proxy_pass http://127.0.0.1:8501;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

##### 4.3.6 HTTPS 证书配置

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d paralegal.yourdomain.com

# 测试自动续期
sudo certbot renew --dry-run
```

#### 方案二: Milvus 独立部署（生产推荐）

对于大规模使用，建议部署独立的 Milvus 集群:

```yaml
# docker-compose.yml 增加 Milvus 服务
  milvus:
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - "etcd"
      - "minio"

  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

volumes:
  milvus_data:
  etcd_data:
  minio_data:
```

修改配置连接独立 Milvus:

```python
# config/settings.py
MILVUS_HOST = "milvus"
MILVUS_PORT = 19530
```

### 4.4 系统服务配置

```ini
# /etc/systemd/system/paralegal-agent.service
[Unit]
Description=Paralegal Agent Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/paralegal-agent
Environment="PATH=/opt/paralegal-agent/.venv/bin"
Environment="OPENAI_API_KEY=your_key"
Environment="FIRECRAWL_API_KEY=your_key"
ExecStart=/opt/paralegal-agent/.venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable paralegal-agent
sudo systemctl start paralegal-agent

# 查看状态
sudo systemctl status paralegal-agent
```

### 4.5 验证部署

1. **访问应用**: 打开浏览器访问 `https://paralegal.yourdomain.com`
2. **上传测试 PDF**: 上传一个测试 PDF 文档
3. **测试问答**: 输入问题，验证回答质量和引用展示
4. **测试网页搜索**: 问一个文档外的问题，验证网页搜索 fallback

### 4.6 监控与日志

```bash
# 查看应用日志
docker-compose logs -f paralegal-agent

# 查看 Milvus 日志
docker-compose logs -f milvus

# 查看系统资源使用
docker stats
```

---

## 5. 功能特性

| 特性 | 说明 |
|------|------|
| 📄 **PDF 文档分析** | 支持上传 PDF 法律文档，自动解析建库 |
| 🔍 **两阶段检索** | 二进制粗过滤 + 精确重排序，又快又准 |
| 📚 **RAG 问答** | 基于文档内容的智能问答，带引用溯源 |
| 🌐 **网页搜索增强** | 文档外问题自动触发 Firecrawl 网页搜索 |
| ⚖️ **质量评估路由** | LLM 自动评估回答质量，决定是否补充搜索 |
| 📊 **工作流可视化** | 展示完整工作流执行日志，便于调试 |
| 🔒 **本地部署** | 支持 Ollama 本地模型，数据不离开服务器 |
| ⚡ **二进制量化** | 向量二值化，内存占用小，检索速度快 |

---

## 6. 常见问题

**Q: 支持哪些文件格式？**
A: 当前主要支持 PDF 格式，可通过扩展支持 Word、TXT 等格式。

**Q: 可以使用本地大模型吗？**
A: 可以，配置 Ollama 即可使用本地模型，数据完全本地化。

**Q: Milvus 本地模式和集群模式有什么区别？**
A: 本地模式适合开发和小规模使用，集群模式适合生产环境，支持水平扩展。

**Q: 如何提高回答准确率？**
A: 可以从三方面优化：1) 提高文档切分质量；2) 调整 Top-K 参数；3) 使用更强的嵌入模型。

**Q: 网页搜索功能是必需的吗？**
A: 不是，没有 Firecrawl API Key 时，系统会自动跳过网页搜索，仅使用文档 RAG。

---

## License

MIT License
