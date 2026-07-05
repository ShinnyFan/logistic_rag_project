import os
from pathlib import Path

# ==================== 项目根目录 ====================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ==================== 数据路径 ====================
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
BENCHMARK_DIR = DATA_DIR / "benchmark"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
CHUNKS_METADATA_PATH = PROCESSED_DIR / "chunks_metadata.pkl"

FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
ID_MAP_PATH = INDEX_DIR / "id_map.pkl"

TEST_SET_PATH = BENCHMARK_DIR / "test_set.json"

# ==================== 输出路径 ====================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
EVAL_RESULTS_PATH = OUTPUTS_DIR / "eval_results.csv"
ANSWERS_PATH = OUTPUTS_DIR / "answers.json"

# ==================== 日志路径 ====================
LOGS_DIR = PROJECT_ROOT / "logs"

# ==================== Qwen API 配置 ====================
# 阿里云百炼平台 API Key（优先从环境变量读取）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 生成模型配置
QWEN_GENERATION_MODEL = "qwen-plus"      # 用于RAG问答生成
QWEN_JUDGE_MODEL = "qwen-max"           # 用于评测裁判（也可用 gpt-4o-mini）
QWEN_EMBEDDING_MODEL = "text-embedding-v3"  # 备用：Qwen API Embedding

# 生成参数
GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_TOKENS = 2048

# ==================== 本地 Embedding 模型配置 ====================
# 使用 sentence-transformers 本地模型
LOCAL_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBEDDING_DIM = 384
LOCAL_EMBEDDING_DEVICE = "cpu"  # "cpu" 或 "cuda"

# Embedding 模式选择："local" 使用本地模型，"api" 使用 Qwen API
EMBEDDING_MODE = "local"

# ==================== 文本切块配置 ====================
CHUNK_SIZE = 512          # 每个chunk的最大token数（近似字符数）
CHUNK_OVERLAP = 128       # chunk之间的重叠token数

# ==================== 检索配置 ====================
RETRIEVAL_TOP_K = 5       # 检索返回的Top-K文本块数

# ==================== 评测配置 ====================
EVAL_SCORE_RANGE = (1, 5)  # 评分范围 1-5 分