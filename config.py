import logging
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 读取项目根目录的 .env（DASHSCOPE_API_KEY / OPENAI_API_KEY 等），
# 这样运行 ingest.py / chat.py 时无需手动 source。
load_dotenv()

# ── Paths ──────────────────────────────────────────────────
DATA_DIR = Path("data")
# DB_DIR 在下面按 CHUNK_SIZE 自动命名（见 Ingestion 段）

# ── Local models（embedding + reranker；跨平台可移植）──
# 解析优先级：环境变量 > 本机已存在的本地路径 > HuggingFace 模型ID(自动下载)。
# → Mac 上继续用 ~/models 里的本地模型(不重下)；Windows/服务器上没本地路径 → 自动从 HF 下载。
def _resolve_model(env_name, local_default, hf_id):
    v = os.environ.get(env_name)
    if v:
        return v
    return local_default if Path(local_default).exists() else hf_id

EMBED_MODEL    = _resolve_model("EMBED_MODEL",    "/Users/taowang/models/bge-m3",              "BAAI/bge-m3")
RERANKER_MODEL = _resolve_model("RERANKER_MODEL", "/Users/taowang/models/bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3")

# 嵌入后端：auto(Apple Silicon→MLX，其余→sentence-transformers/torch) | mlx | st。环境变量 EMBED_BACKEND 可覆盖。
EMBED_BACKEND  = os.environ.get("EMBED_BACKEND", "auto")

# ── LLM / VL provider（可切换）─────────────────────────────
# 一键换模型：编辑 models.yaml 里的 active（openai/claude/gemini/qwen/local）。
# 这里只负责把 yaml 读进来；具体调用在 model_client.py。
MODELS = yaml.safe_load((Path(__file__).parent / "models.yaml").read_text(encoding="utf-8"))

def _active(role: str) -> dict:
    """返回某个角色（'chat' 或 'vl'）当前生效的 provider 配置。"""
    override = MODELS.get("llm_override") if role == "chat" else MODELS.get("vl_override")
    name = override or MODELS["active"]
    return name, MODELS["providers"][name]

# 向后兼容：旧代码若 import 这两个名字仍可用（取当前 active 的模型名）。
LLM_MODEL = _active("chat")[1]["chat_model"]
VL_MODEL  = _active("vl")[1]["vl_model"]

def ocr_model() -> str:
    """扫描页/表格“转录”用的 OCR 模型名。

    当前 VL provider 若配了 ocr_model（如 qwen 的 qwen-vl-ocr）就用它；没配则回退到该
    provider 的 vl_model —— 这样换到别的 provider（openai/claude…）不会因缺 ocr_model 报错，
    “一键换 provider” 仍成立。图片“理解”（describe_figure）不走这里，仍用 vl_model。
    """
    _name, spec = _active("vl")
    return spec.get("ocr_model") or spec["vl_model"]

# ── Ingestion ───────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 150
BATCH_SIZE    = 64

# 索引文件夹按 chunk_size 自动分开：500→index_chunk500/  1000→index_chunk1000/
# 只改 CHUNK_SIZE 就同时决定“切多大”和“存/读哪个索引”。vl_cache/ 不在这里，全版本共享。
DB_DIR = Path(f"index_chunk{CHUNK_SIZE}")

# ── Retrieval ───────────────────────────────────────────────
CANDIDATE_K = 100
MIN_K       = 3

# ── Chat ────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
MAX_TOKENS        = 800

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
