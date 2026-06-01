<!-- README.md -->
# ArXiv 检索对比 Demo

快速运行（建议在虚拟环境中）:

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

说明：
- 在侧边栏选择算法（TF-IDF 或 BM25），输入查询并点击搜索。
- 内置测试数据为 `data/sample_arxiv_20000.jsonl`。
- 如果要用你自己的数据，请放置 JSONL 文件到 `data/` 目录，格式与 `data/sample_arxiv_20000.jsonl` 类似（每行一个 JSON 对象，包含 `id`, `title`, `abstract`, `authors`, `categories`, `update_date`）。

**如何使用检索系统**

- **先决条件:** Python 3.8+，在项目根目录下推荐使用虚拟环境。依赖在 [requirements.txt](requirements.txt) 中列出。
- **一键准备（Windows PowerShell）:** 推荐运行一键脚本来创建虚拟环境、安装依赖并导入示例数据：

```powershell
powershell -ExecutionPolicy ByPass -File "scripts/setup_env.ps1"
```

- **手动步骤（可选）:**

```bash
# 创建并激活虚拟环境（Windows PowerShell 示例）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 可选：初始化数据库（尝试 MySQL，失败则回退到 SQLite）
python database/create_database.py

# 将示例 JSONL 导入数据库（或指定你自己的 JSONL）
python database/import_data.py --input data/sample_arxiv_20000.jsonl --limit 20000 --batch 500
```

- **运行 Demo（Streamlit）:**

```bash
streamlit run app_streamlit.py --server.port 8502
```

- **在应用中切换算法:**
	- 启动后在侧边栏选择算法：TF-IDF / BM25 / Sentence-BERT（SBERT）。
	- TF-IDF 与 BM25 在线构建索引；SBERT 第一次运行会从 Hugging Face 下载模型并计算嵌入，可能会较慢并占用更多内存。

**基准测试与算法说明**

- 我已针对 1k 与 10k 文档规模运行快速基准，结果保存为 `data/benchmarks_summary.csv`。关键结果（50 queries/sample）：
	- 1k:
		- TF-IDF: 构建 0.485s，平均查询 0.0021s
		- BM25:  构建 0.089s，平均查询 0.0058s
		- SBERT: 构建 60.58s，平均查询 0.0269s
	- 10k:
		- TF-IDF: 构建 8.37s，平均查询 0.0426s
		- BM25:  构建 1.87s，平均查询 0.1396s
		- SBERT: 构建 377.79s，平均查询 0.0285s

- 算法实现要点：
	- TF-IDF: 使用 `sklearn.feature_extraction.text.TfidfVectorizer`，默认参数中使用 `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`，向量余弦相似度使用 `linear_kernel` 计算。
	- BM25: 使用 `rank_bm25.BM25Okapi`，对 `title_abstract` 字段先做 `tokenize_and_filter()`（去停用词、短词过滤），然后基于 BM25 分数检索并可选归一化到 [0,1]。
	- SBERT: 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成句向量（`normalize_embeddings=True`），当前实现为暴力向量点积检索；我已在代码中添加可选的 FAISS HNSW 接口用于大规模加速（构建/加载/搜索方法位于 `search_backend.py`）。

- 注意事项：SBERT 首次构建会下载模型并花费较长时间，建议在生产或大规模实验中先构建并保存嵌入与 FAISS 索引以加速查询。


- **关于 SBERT:**
	- 推荐模型: `sentence-transformers/all-MiniLM-L6-v2`（默认）。首次使用需联网下载模型权重。
	- 若机器内存受限，可优先使用 TF-IDF 或 BM25。

- **关于数据库（可选 MySQL）:**
	- 如需连接 MySQL，请通过环境变量设置连接信息：`ARXIV_DB_HOST`、`ARXIV_DB_USER`、`ARXIV_DB_PASSWORD`、`ARXIV_DB_NAME`、`ARXIV_DB_CHARSET`。脚本会尝试使用 `pymysql` 连接，失败时自动回退到本地 SQLite（`data/arxiv.db`）。
	- 大模型接口也建议使用环境变量配置：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`。默认接口地址为 `https://api.deepseek.com`，默认模型为 `deepseek-v4-pro`。

- **示例文件与位置:**
	- 应用入口: [app_streamlit.py](app_streamlit.py)
	- 后端检索实现: [search_backend.py](search_backend.py)
	- 数据库相关: [database/create_database.py](database/create_database.py), [database/import_data.py](database/import_data.py)
	- 示例数据: [data/sample_arxiv_20000.jsonl](data/sample_arxiv_20000.jsonl)（默认内置），SQLite 文件: [data/arxiv.db](data/arxiv.db)

**推荐的检索逻辑（实践建议）**

- **两阶段检索（推荐，准确且高效）**: 先用轻量级的倒排/稀疏检索（BM25 或 TF‑IDF）快速筛出候选集（如 top‑100），再用 SBERT 向量对候选集重排（re‑ranking），兼顾召回与语义精度且节省向量计算成本。
- **全量向量检索（低延迟、大规模）**: 对于需要在几十万或更多文档上低延迟返回，先用 SBERT 生成并持久化嵌入，再用 FAISS（HNSW 或 IVF+PQ）做 ANN 检索；HNSW 简单易上手，IVF+PQ 在超大规模时更节省空间。
- **缓存与持久化**: 持久化 `tfidf` 向量化器（`joblib.dump`）和稀疏矩阵（`scipy.sparse.save_npz`），以及 SBERT 嵌入（`.npy`）和 FAISS 索引（`.index`），启动时按需加载以减少重复构建成本。
- **工程实践**: 将批量嵌入/索引构建作为离线任务（cron / CI），查询服务只做加载索引与快速检索；当资源紧张时优先使用 BM25/TF‑IDF 在线搜索并在后台异步构建向量索引。

**仓库中主要文件说明**

- [app_streamlit.py](app_streamlit.py): 前端演示（Streamlit），算法切换、查询界面与结果展示。
- [search_backend.py](search_backend.py): 后端检索核心，实现 TF‑IDF、BM25、SBERT（暴力向量检索）与可选 FAISS（HNSW）构建/加载/搜索接口。
- [run_quick_benchmarks.py](run_quick_benchmarks.py): 快速基准测试脚本（1k/10k 测试）。
- [run_end2end_test.py](run_end2end_test.py): 端到端验证脚本（初始化检索、示例查询、SBERT 子集测试）。
- [data/benchmarks_summary.csv](data/benchmarks_summary.csv): 本次基准结果摘要（1k / 10k）。
- [data/sample_arxiv_20000.jsonl](data/sample_arxiv_20000.jsonl): 用于测试的计算机领域 20k 抽样数据（演示用）。
- [data/arxiv-metadata-oai-snapshot.json](data/arxiv-metadata-oai-snapshot.json): 原始完整快照（未经全部处理，大文件）。
- [database/create_database.py](database/create_database.py): 初始化数据库（优先 MySQL，失败回退 SQLite）并创建 `papers` 表。
- [database/import_data.py](database/import_data.py): 统一的导入脚本，支持 `--to auto|mysql|sqlite|both`，批量导入 JSONL 到数据库并做字段清洗/映射。
- [database/db_config.py](database/db_config.py): MySQL 连接配置（本地示例，部署时请修改凭据）。
- [scripts/setup_env.ps1](scripts/setup_env.ps1): 一键准备脚本（创建虚拟环境、安装依赖、初始化 DB、导入示例数据）。
- [requirements.txt](requirements.txt): Python 依赖列表。


- **常见问题 / 排错:**
	- 如果 Streamlit 页面未显示结果，确认虚拟环境已激活且依赖安装正确；在项目根目录执行 `pip install -r requirements.txt`。
	- SBERT 报错通常与内存或模型下载失败有关，检查网络与可用内存；可先用 TF-IDF 或 BM25 进行演示。
	- 如果导入数据过程中出现 MySQL 权限错误，脚本会提示并回退到 SQLite；如需 MySQL 请检查 `database/db_config.py` 的用户名/密码与数据库权限。

