# search_backend.py
import json
import re
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

# 全局停用词集合
ENGLISH_STOP_WORDS = {
    # 基础停用词
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
    'this', 'these', 'those', 'such', 'there', 'their', 'them',
    'can', 'may', 'might', 'could', 'would', 'should',
    
    # 学术论文常见停用词
    'using', 'used', 'use', 'uses', 'based', 'via', 'through',
    'however', 'therefore', 'thus', 'hence', 'also', 'too',
    'very', 'really', 'quite', 'rather', 'some', 'any',
    'about', 'against', 'between', 'into', 'throughout', 'during',
    'without', 'within', 'along', 'following', 'including',
    'et', 'al', 'eg', 'ie', 'etc', 'fig', 'figure', 'table',
    
    # 保留否定词（不过滤）
    # 'no', 'not', 'none', 'neither', 'nor' - 不过滤这些
}

# 保留的重要短词
KEEP_SHORT_WORDS = {'no', 'not', 'ai', 'cv', 'nlp', 'ml', '3d', '2d'}


def tokenize_and_filter(text: str, stop_words: Set[str] = ENGLISH_STOP_WORDS) -> List[str]:
    """
    分词并过滤停用词
    """
    if not text:
        return []
    
    # 转小写
    text = text.lower()
    # 去除标点符号（保留字母数字和连字符）
    text = re.sub(r'[^\w\s-]', ' ', text)
    # 分词
    tokens = text.split()
    
    # 过滤：去除停用词和过短的词（除非在保留列表中）
    filtered = []
    for token in tokens:
        if token in stop_words:
            continue
        if len(token) <= 1:
            continue
        if len(token) == 2 and token not in KEEP_SHORT_WORDS:
            continue
        filtered.append(token)
    
    return filtered


def load_jsonl(path: str, max_docs: Optional[int] = None) -> List[Dict]:
    """
    加载 JSONL 格式的数据文件
    """
    docs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    return docs


class SearchEngine:
    """论文检索引擎，支持 TF-IDF、BM25 和 Sentence-BERT"""
    
    def __init__(self, docs: List[Dict], text_field: str = 'title_abstract',
                 use_stopwords: bool = True, auto_build: bool = False):
        """
        初始化搜索引擎
        
        Args:
            docs: 论文文档列表
            text_field: 用于检索的文本字段名
            use_stopwords: 是否使用停用词过滤
            auto_build: 是否自动构建所有索引
        """
        self.docs = docs
        self.text_field = text_field
        self.use_stopwords = use_stopwords
        
        # 构建语料库
        self.corpus = [d.get(text_field, '') for d in docs]
        
        # 算法标志
        self._tfidf_built = False
        self._bm25_built = False
        self._sbert_built = False
        
        # TF-IDF
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # BM25
        self.bm25 = None
        self.tokenized_corpus = None
        
        # Sentence-BERT
        self.sbert_model = None
        self.sbert_embeddings = None
        
        # 自动构建
        if auto_build:
            self.build_tfidf()
            self.build_bm25()
    
    def _check_index(self, doc_index: int) -> None:
        """检查文档索引是否有效"""
        if doc_index < 0 or doc_index >= len(self.docs):
            raise IndexError(f"文档索引 {doc_index} 超出范围 [0, {len(self.docs)})")
    
    # ==================== TF-IDF ====================
    
    def build_tfidf(self, max_features: int = 50000, ngram_range: tuple = (1, 2)):
        """
        构建 TF-IDF 索引
        
        Args:
            max_features: 最大特征数
            ngram_range: n-gram 范围
        """
        if self._tfidf_built:
            return
        
        print(f"正在构建 TF-IDF 索引，语料库大小: {len(self.corpus)}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english' if self.use_stopwords else None,
            ngram_range=ngram_range,
            min_df=2,  # 忽略出现次数少于2的词
            max_df=0.95  # 忽略出现在95%以上文档中的词
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self._tfidf_built = True
        
        print(f"TF-IDF 索引构建完成，维度: {self.tfidf_matrix.shape}")
    
    def search_tfidf(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        使用 TF-IDF 进行检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        if not query or not query.strip():
            return []
        
        if not self._tfidf_built:
            self.build_tfidf()
        
        # 转换查询
        q_vec = self.vectorizer.transform([query])
        # 计算相似度
        sims = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        # 获取 top_k
        top_indices = np.argsort(-sims)[:top_k]
        
        results = []
        for i in top_indices:
            if sims[i] > 0:  # 只返回正相关结果
                results.append((int(i), float(sims[i]), self.docs[i]))
        
        return results
    
    def get_similar_tfidf(self, doc_index: int, top_k: int = 5) -> List[tuple]:
        """
        使用 TF-IDF 查找相似文档
        
        Args:
            doc_index: 文档索引
            top_k: 返回结果数量
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        self._check_index(doc_index)
        
        if not self._tfidf_built:
            self.build_tfidf()
        
        # 获取文档向量
        doc_vec = self.tfidf_matrix[doc_index]
        # 计算相似度
        sims = linear_kernel(doc_vec, self.tfidf_matrix).flatten()
        # 排除自身
        sims[doc_index] = -1
        # 获取 top_k
        top_indices = np.argsort(-sims)[:top_k]
        
        return [(int(i), float(sims[i]), self.docs[i]) for i in top_indices if sims[i] > 0]
    
    # ==================== BM25 ====================
    
    def build_bm25(self):
        """构建 BM25 索引"""
        if self._bm25_built:
            return
        
        try:
            from rank_bm25 import BM25Okapi as _BM25
        except ImportError:
            raise RuntimeError('rank_bm25 未安装，请运行: pip install rank_bm25')
        
        print(f"正在构建 BM25 索引，语料库大小: {len(self.corpus)}")
        
        # 分词并过滤停用词
        self.tokenized_corpus = []
        for doc in self.corpus:
            if self.use_stopwords:
                tokens = tokenize_and_filter(doc)
            else:
                tokens = doc.lower().split()
            self.tokenized_corpus.append(tokens)
        
        self.bm25 = _BM25(self.tokenized_corpus)
        self._bm25_built = True
        
        print(f"BM25 索引构建完成")
    
    def search_bm25(self, query: str, top_k: int = 10, normalize: bool = True) -> List[tuple]:
        """
        使用 BM25 进行检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            normalize: 是否归一化分数到 [0, 1]
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        if not query or not query.strip():
            return []
        
        if not self._bm25_built:
            self.build_bm25()
        
        # 处理查询
        if self.use_stopwords:
            tokenized_query = tokenize_and_filter(query)
        else:
            tokenized_query = query.lower().split()
        
        # 如果查询为空，返回空结果
        if not tokenized_query:
            return []
        
        # 计算分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化
        if normalize and len(scores) > 0:
            max_score = scores.max()
            min_score = scores.min()
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
            elif max_score > 0:
                scores = scores / max_score
        
        # 获取 top_k
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append((int(i), float(scores[i]), self.docs[i]))
        
        return results
    
    def get_similar_bm25(self, doc_index: int, top_k: int = 5, normalize: bool = True) -> List[tuple]:
        """
        使用 BM25 查找相似文档
        
        Args:
            doc_index: 文档索引
            top_k: 返回结果数量
            normalize: 是否归一化分数
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        self._check_index(doc_index)
        
        if not self._bm25_built:
            self.build_bm25()
        
        # 获取文档 tokens
        doc_tokens = self.tokenized_corpus[doc_index]
        
        if not doc_tokens:
            return []
        
        # 计算分数
        scores = self.bm25.get_scores(doc_tokens)
        
        # 归一化
        if normalize and len(scores) > 0:
            max_score = scores.max()
            min_score = scores.min()
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
            elif max_score > 0:
                scores = scores / max_score
        
        # 排除自身
        scores[doc_index] = -1
        # 获取 top_k
        top_indices = np.argsort(-scores)[:top_k]
        
        return [(int(i), float(scores[i]), self.docs[i]) for i in top_indices if scores[i] > 0]
    
    # ==================== Sentence-BERT ====================
    
    def build_sbert(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                    batch_size: int = 32):
        """
        构建 Sentence-BERT 索引
        
        Args:
            model_name: SBERT 模型名称
            batch_size: 批处理大小
        """
        if self._sbert_built:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError('sentence-transformers 未安装，请运行: pip install sentence-transformers')
        
        print(f"正在加载 SBERT 模型: {model_name}")
        self.sbert_model = SentenceTransformer(model_name)
        
        print(f"正在编码 {len(self.corpus)} 篇文档...")
        # 使用批处理编码，显示进度
        embeddings = self.sbert_model.encode(
            self.corpus,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size
        )
        self.sbert_embeddings = np.array(embeddings, dtype=np.float32)
        self._sbert_built = True
        
        print(f"SBERT 索引构建完成，嵌入维度: {self.sbert_embeddings.shape}")
    
    def search_sbert(self, query: str, top_k: int = 10,
                     model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> List[tuple]:
        """
        使用 Sentence-BERT 进行语义检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            model_name: SBERT 模型名称
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        if not query or not query.strip():
            return []
        
        if not self._sbert_built:
            self.build_sbert(model_name=model_name)
        
        # 编码查询
        q_emb = self.sbert_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        # 计算相似度（点积）
        sims = np.dot(self.sbert_embeddings, q_emb)
        
        # 获取 top_k
        top_indices = np.argsort(-sims)[:top_k]
        
        results = []
        for i in top_indices:
            if sims[i] > 0:
                results.append((int(i), float(sims[i]), self.docs[i]))
        
        return results
    
    def get_similar_sbert(self, doc_index: int, top_k: int = 5,
                          model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> List[tuple]:
        """
        使用 Sentence-BERT 查找语义相似文档
        
        Args:
            doc_index: 文档索引
            top_k: 返回结果数量
            model_name: SBERT 模型名称
        
        Returns:
            [(索引, 相似度分数, 文档), ...]
        """
        self._check_index(doc_index)
        
        if not self._sbert_built:
            self.build_sbert(model_name=model_name)
        
        # 获取文档嵌入
        doc_emb = self.sbert_embeddings[doc_index]
        # 计算相似度
        sims = np.dot(self.sbert_embeddings, doc_emb)
        # 排除自身
        sims[doc_index] = -1
        # 获取 top_k
        top_indices = np.argsort(-sims)[:top_k]
        
        return [(int(i), float(sims[i]), self.docs[i]) for i in top_indices if sims[i] > 0]

    # ================ SBERT + FAISS (ANN) ================
    def build_sbert_faiss(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                          batch_size: int = 32,
                          index_path: str = 'data/faiss.index',
                          emb_path: str = 'data/sbert_embeddings.npy',
                          hnsw_m: int = 32,
                          ef_construction: int = 200):
        """
        使用 SBERT 生成嵌入并用 FAISS (HNSW) 构建 ANN 索引，索引与嵌入会被持久化到磁盘。
        """
        import os
        try:
            import faiss
        except Exception:
            raise RuntimeError('faiss 未安装，请运行: pip install faiss-cpu（或 faiss）')

        # 确保有嵌入
        if not self._sbert_built or self.sbert_embeddings is None:
            self.build_sbert(model_name=model_name, batch_size=batch_size)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        # 保存嵌入
        np.save(emb_path, self.sbert_embeddings)

        d = int(self.sbert_embeddings.shape[1])
        # 使用 HNSW 构建近邻图（在归一化向量上使用 L2 距离）
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        try:
            index.hnsw.efConstruction = ef_construction
        except Exception:
            pass

        # 将向量添加到索引（FAISS 接受 float32）
        index.add(self.sbert_embeddings.astype('float32'))

        # 持久化索引
        faiss.write_index(index, index_path)

        # 保存索引路径信息在实例变量
        self._faiss_index_path = index_path
        self._faiss_emb_path = emb_path
        self._faiss_index = index

        print(f"FAISS HNSW 索引已构建并保存: {index_path}")

    def load_sbert_faiss(self, index_path: str = 'data/faiss.index',
                         emb_path: str = 'data/sbert_embeddings.npy') -> None:
        """
        从磁盘加载 FAISS 索引与 SBERT 嵌入（如果存在）
        """
        try:
            import faiss
        except Exception:
            raise RuntimeError('faiss 未安装，请运行: pip install faiss-cpu（或 faiss）')

        import os
        if not os.path.exists(index_path) or not os.path.exists(emb_path):
            raise FileNotFoundError('指定的索引文件或嵌入文件不存在')

        index = faiss.read_index(index_path)
        emb = np.load(emb_path)

        self.sbert_embeddings = emb.astype('float32')
        self._sbert_built = True
        self._faiss_index = index
        self._faiss_index_path = index_path
        self._faiss_emb_path = emb_path

        print(f"已从磁盘加载 FAISS 索引: {index_path}")

    def search_sbert_faiss(self, query: str, top_k: int = 10,
                           model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                           index_path: str = 'data/faiss.index',
                           emb_path: str = 'data/sbert_embeddings.npy') -> List[tuple]:
        """
        使用 FAISS 索引进行 SBERT 语义检索（若索引未加载则尝试从磁盘加载）
        """
        if not query or not query.strip():
            return []

        try:
            import faiss
        except Exception:
            raise RuntimeError('faiss 未安装，请运行: pip install faiss-cpu（或 faiss）')

        # 如果索引尚未加载，尝试从磁盘加载或构建
        if not hasattr(self, '_faiss_index') or self._faiss_index is None:
            import os
            if os.path.exists(index_path) and os.path.exists(emb_path):
                self.load_sbert_faiss(index_path=index_path, emb_path=emb_path)
            else:
                # 构建并保存索引
                self.build_sbert_faiss(model_name=model_name, index_path=index_path, emb_path=emb_path)

        # 编码查询向量
        if not self._sbert_built or self.sbert_embeddings is None:
            # 保底再构建一次
            self.build_sbert(model_name=model_name)

        q_emb = self.sbert_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        q_vec = np.array([q_emb], dtype='float32')

        # 使用 FAISS 搜索
        D, I = self._faiss_index.search(q_vec, top_k)

        results = []
        for idx in I[0]:
            if idx < 0:
                continue
            # 计算更直观的相似度分数（点积）
            score = float(np.dot(self.sbert_embeddings[idx], q_emb))
            if score > 0:
                results.append((int(idx), score, self.docs[idx]))

        return results
    
    # ==================== 工具方法 ====================
    
    def get_doc_by_index(self, index: int) -> Optional[Dict]:
        """根据索引获取文档"""
        if 0 <= index < len(self.docs):
            return self.docs[index]
        return None
    
    def get_doc_by_id(self, doc_id: str) -> Optional[Dict]:
        """根据论文ID获取文档"""
        for doc in self.docs:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def get_stats(self) -> Dict:
        """获取搜索引擎统计信息"""
        return {
            'num_docs': len(self.docs),
            'tfidf_built': self._tfidf_built,
            'bm25_built': self._bm25_built,
            'sbert_built': self._sbert_built,
            'use_stopwords': self.use_stopwords,
            'text_field': self.text_field
        }


def make_title_abstract_field(raw_docs: List[Dict]) -> List[Dict]:
    """
    为每篇文档创建 title_abstract 组合字段
    """
    out = []
    for d in raw_docs:
        title = d.get('title') or ''
        abstract = d.get('abstract') or ''
        combined = f"{title}\n{abstract}"
        new = dict(d)
        new['title_abstract'] = combined
        out.append(new)
    return out


def load_from_mysql(max_docs: Optional[int] = None) -> List[Dict]:
    """从 MySQL 数据库加载论文，返回与 load_jsonl 相同格式的 dict 列表"""
    import pymysql
    from database.db_config import DB_CONFIG

    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
    )
    cursor = conn.cursor()

    limit_clause = f"LIMIT {max_docs}" if max_docs and max_docs > 0 else ""
    sql = f"SELECT id, title, abstract, authors, categories, update_date, doi FROM papers {limit_clause}"
    cursor.execute(sql)

    columns = ["id", "title", "abstract", "authors", "categories", "update_date", "doi"]
    docs = []
    for row in cursor.fetchall():
        doc = dict(zip(columns, row))
        if doc["update_date"] and hasattr(doc["update_date"], "strftime"):
            doc["update_date"] = doc["update_date"].strftime("%Y-%m-%d")
        docs.append(doc)

    cursor.close()
    conn.close()
    return docs


def load_from_sqlite(db_path: str = "data/arxiv.db", max_docs: Optional[int] = None) -> List[Dict]:
    """从本地 SQLite 数据库加载论文，字段格式与 MySQL/JSONL 保持一致。"""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite 数据库文件不存在: {db_path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        limit_clause = "LIMIT ?" if max_docs and max_docs > 0 else ""
        sql = f"SELECT id, title, abstract, authors, categories, update_date, doi FROM papers {limit_clause}"
        rows = conn.execute(sql, (int(max_docs),) if limit_clause else ()).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_from_database(max_docs: Optional[int] = None, sqlite_path: str = "data/arxiv.db") -> List[Dict]:
    """优先读取 MySQL；连接失败时自动回退到随项目提交的 SQLite 数据库。"""
    try:
        return load_from_mysql(max_docs=max_docs)
    except Exception as exc:
        logger.warning("MySQL 加载失败，回退到 SQLite: %s", exc)
        return load_from_sqlite(sqlite_path, max_docs=max_docs)
