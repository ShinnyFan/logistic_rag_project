# 物流调度领域 RAG 问答系统与自动化评测框架

面向物流调度研究员/工程师的 AI 助手，能精准回答关于车辆路径问题（VRP）、自适应大邻域搜索（ALNS）、动态调度、车货匹配等专业问题。

## 技术架构

```
[PDF/TXT文件夹] → (pdf_loader) → 原始文本 → (chunker) → Chunks
Chunks → (embedder + vector_store) → FAISS索引 + id映射

[用户问题] → (retriever) → Top-K Chunks
Top-K Chunks + [用户问题] → (prompt组装) → Qwen API → 最终回答

[测试集JSON] → (evaluator) → 调用RAG Pipeline获取回答 → 调用裁判模型打分 → CSV结果
CSV结果 → (analyzer) → 统计图表 & 低分清单
```

## 项目结构

```
logistics_rag_project/
├── config.py                 # 全局配置
├── requirements.txt          # 依赖清单
├── main.py                   # 统一命令行入口
├── .env.example              # 环境变量示例
│
├── data/
│   ├── raw_pdfs/             # 知识文档（PDF/TXT）
│   ├── processed/            # 中间产物（chunks.json）
│   ├── index/                # FAISS索引持久化
│   └── benchmark/            # 评测数据集
│       └── test_set.json     # 10道物流领域典型问题
│
├── src/
│   ├── pdf_loader.py         # 文档加载器（PDF + TXT）
│   ├── chunker.py            # 文本切块
│   ├── embedder.py           # Embedding模型
│   ├── vector_store.py       # FAISS向量存储
│   ├── retriever.py          # 检索逻辑
│   ├── prompt_templates.py   # Prompt模板
│   ├── qa_chain.py           # Qwen API调用
│   ├── rag_pipeline.py       # RAG流程编排
│   ├── evaluator.py          # 自动化评测
│   └── analyzer.py           # 结果分析与可视化
│
├── outputs/                  # 评测输出
│   ├── eval_results.csv
│   └── charts/
│
├── logs/                     # 日志
└── tests/                    # 单元测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置阿里云百炼 API Key：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-api-key"

# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key"
```

或复制 `.env.example` 为 `.env` 并填入密钥。

### 3. 准备知识库

将物流领域的 PDF 或 TXT 文档放入 `data/raw_pdfs/` 目录。项目已包含示例知识文本 `knowledge_sample.txt`。

### 4. 构建索引

```bash
python main.py build-index
```

这将依次执行：文档加载 → 文本切块 → Embedding生成 → FAISS索引构建。

### 5. 交互式问答

```bash
python main.py ask
```

输入问题即可获得基于知识库的 RAG 回答。

### 6. 运行评测

```bash
# 基础评测
python main.py eval

# 评测 + 消融实验（RAG vs 纯Qwen）
python main.py eval --ablation

# 评测 + 自动分析
python main.py eval --analyze
```

### 7. 分析结果

```bash
python main.py analyze --export-low
```

## 评测维度

| 维度 | 说明 | 评分范围 |
|------|------|----------|
| 相关性 | 是否切题，有无答非所问 | 1-5 |
| 事实一致性 | 是否与公认物流知识冲突或存在幻觉 | 1-5 |
| 完整性 | 是否覆盖参考答案中的关键点 | 1-5 |

## 技术选型

| 模块 | 选型 | 理由 |
|------|------|------|
| 向量库 | FAISS (IndexFlatIP) | 纯本地存储，无网络依赖 |
| Embedding | sentence-transformers (MiniLM) | 本地推理，384维，CPU高速 |
| 文本切块 | LangChain RecursiveCharacterTextSplitter | 语义保留，递归切割 |
| 文档解析 | pdfplumber + pypdf | 双保险，表格+文本 |
| 生成模型 | Qwen API (兼容OpenAI) | 中文优秀，SDK统一 |
| 评测裁判 | Qwen-max | CoT思维链评分 |
| 分析工具 | Pandas + Matplotlib/Seaborn | 标准数据科学生态 |

## 测试集

`data/benchmark/test_set.json` 包含 20 道物流领域典型问题，覆盖：

- VRP基础概念
- VRP求解算法
- ALNS算法
- 动态调度
- 车货匹配
- 综合应用
 
## 输出示例

### eval_results.csv

| id | category | question | relevance | consistency | completeness | overall |
|----|----------|----------|-----------|-------------|--------------|---------|
| T001 | VRP基础概念 | 什么是车辆路径问题... | 5 | 4 | 4 | 4.3 |
| T004 | ALNS算法 | ALNS算法中... | 5 | 5 | 4 | 4.7 |

### 分析图表（outputs/charts/）

- `score_distribution.png` - 各维度评分分布直方图
- `category_radar.png` - 各分类评分雷达图
- `retrieval_score_distribution.png` - 检索相似度分布
- `ablation_boxplot.png` - 消融实验箱线图
- `ablation_comparison.png` - 消融实验各维度对比

