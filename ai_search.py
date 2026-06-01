"""
AI 检索引擎 — 基于 LLM 的查询增强、RAG 答案生成、对话追问
支持 OpenAI 兼容 API（DeepSeek / OpenAI / Ollama 等）
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"


@dataclass
class AISearchConfig:
    api_key: str = ""
    base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL))
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout_seconds: float = 25.0

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_key = (self.api_key or "").strip().rstrip("/")
        if not self.base_url:
            self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        self.base_url = (self.base_url or "").strip().rstrip("/")
        if not self.model:
            self.model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.model = (self.model or "").strip()


QUERY_ENHANCE_SYSTEM = """你是一个学术论文检索专家。用户会用自然语言描述他们想了解的研究方向或问题。

你的任务是将用户的查询改写为适合论文检索引擎（TF-IDF / BM25 / 语义向量检索）的英文关键词组合。

要求：
- 关键词必须是英文（即使用户用中文提问）
- 精准覆盖核心概念，包含同义词和相关术语
- 去除口语化的冗余表达，保留技术术语
- 长度控制在 3-8 个词或短语

严格按 JSON 格式输出，不要输出其他内容：
{"keywords": "改写后的英文关键词", "explanation": "简短中文说明改写的思路"}"""


RAG_SYSTEM = """你是一个专业的学术论文分析助手。你会收到一个用户问题以及若干篇检索到的相关论文（含标题、作者、摘要等信息）。

请基于这些论文的内容，用中文回答用户的问题。

要求：
- 引用具体论文时标注 [编号]，如 [1]、[2]
- 综合多篇论文的信息，不要只引用单篇
- 如果论文之间存在观点或方法的差异，请指出对比
- 使用结构化的输出（分点、小标题）以便阅读
- 如果检索结果不足以完整回答问题，在末尾诚实说明
- 回答末尾列出「📚 参考文献」编号与标题的对应关系"""


CHAT_SYSTEM = """你是一个专业的学术论文对话助手。你可以访问当前检索到的论文作为知识来源。

对话规则：
- 用中文回复，保持专业但友好的语气
- 回答问题时优先引用上下文中的论文
- 如果上下文中没有相关信息，可以基于你的知识回答但要注明
- 回答简洁，重点突出"""


def _build_client(config: AISearchConfig):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai 未安装，请运行: pip install openai>=1.0.0")

    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_seconds)


def _friendly_error(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return "API Key 认证失败，请检查是否复制完整、是否多了空格或斜杠，或该 Key 是否已失效。"
    if status_code == 429:
        return "接口限流或额度不足，请稍后重试或更换可用 Key。"
    if status_code and 500 <= status_code < 600:
        return "模型服务暂时不可用，请稍后重试。"
    text = str(error)
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "模型接口响应超时，本地检索结果仍可继续查看。"
    return "模型接口暂时不可用，本地检索结果仍可继续查看。"


def _format_papers_context(papers: List[Dict]) -> str:
    """将检索到的论文格式化为 LLM 上下文"""
    lines = []
    for i, doc in enumerate(papers, 1):
        title = doc.get("title", "无标题")
        authors = doc.get("authors", "未知")[:120]
        abstract = doc.get("abstract", "无摘要")[:600]
        year = (doc.get("update_date") or "")[:4]
        lines.append(
            f"[{i}] **{title}**\n"
            f"    作者: {authors}\n"
            f"    年份: {year}\n"
            f"    摘要: {abstract}\n"
        )
    return "\n".join(lines)


# ============================================================
class AISearchEngine:
    """AI 增强检索引擎"""

    def __init__(self, config: Optional[AISearchConfig] = None):
        self.config = config or AISearchConfig()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _build_client(self.config)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    # ==================== 查询增强 ====================

    def enhance_query(self, query: str) -> Tuple[str, str]:
        """
        使用 LLM 增强搜索查询

        Returns:
            (enhanced_keywords, explanation)
        """
        if not self.is_configured:
            return query, ""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": QUERY_ENHANCE_SYSTEM},
                    {"role": "user", "content": query},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            keywords = data.get("keywords", query)
            explanation = data.get("explanation", "")
            return keywords, explanation
        except Exception as e:
            logger.warning(f"查询增强失败: {e}")
            return query, _friendly_error(e)

    # ==================== RAG 答案生成 ====================

    def generate_answer(self, query: str, papers: List[Dict]) -> str:
        """非流式生成答案"""
        if not self.is_configured or not papers:
            return ""

        context = _format_papers_context(papers)
        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": f"检索到的论文：\n\n{context}\n\n用户问题：{query}"},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"RAG 生成失败: {e}")
            return _friendly_error(e)

    def generate_answer_stream(self, query: str, papers: List[Dict]) -> Generator[str, None, None]:
        """流式生成答案"""
        if not self.is_configured or not papers:
            yield ""
            return

        context = _format_papers_context(papers)
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": f"检索到的论文：\n\n{context}\n\n用户问题：{query}"},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"RAG 流式生成失败: {e}")
            yield f"\n\n> {_friendly_error(e)}"

    # ==================== 对话追问 ====================

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]],
        context_papers: List[Dict],
    ) -> str:
        """非流式对话"""
        if not self.is_configured:
            return "请先配置 API Key"

        context = _format_papers_context(context_papers[:5]) if context_papers else ""
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        if context:
            messages.append(
                {"role": "system", "content": f"当前可参考的论文上下文：\n\n{context}"}
            )
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"对话失败: {e}")
            return _friendly_error(e)

    def chat_stream(
        self,
        message: str,
        history: List[Dict[str, str]],
        context_papers: List[Dict],
    ) -> Generator[str, None, None]:
        """流式对话"""
        if not self.is_configured:
            yield "请先配置 API Key"
            return

        context = _format_papers_context(context_papers[:5]) if context_papers else ""
        messages = [{"role": "system", "content": CHAT_SYSTEM}]
        if context:
            messages.append(
                {"role": "system", "content": f"当前可参考的论文上下文：\n\n{context}"}
            )
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.warning(f"对话流式失败: {e}")
            yield f"\n\n> {_friendly_error(e)}"
