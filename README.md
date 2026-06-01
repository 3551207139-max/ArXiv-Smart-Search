# ArXiv Smart Search：论文推荐与检索系统

这是一个面向课程项目的 arXiv 论文检索与推荐系统。项目基于 20000 条 arXiv 论文样例数据，提供关键词检索、语义检索、高级检索、相似论文推荐，以及基于 DeepSeek API 的查询改写、结果综述和追问功能。

## 主要功能

- 多种检索算法：TF-IDF、BM25、Sentence-BERT。
- 高级检索：支持按标题、作者、摘要、分类、日期范围和排除词过滤。
- 论文详情页：展示标题、作者、摘要、分类、更新时间，并给出相似论文推荐。
- 增强检索：可调用 DeepSeek API，将中文自然语言问题改写为英文检索关键词。
- RAG 结果综述：基于 Top 10 检索结果生成简要分析。
- 结果追问：可围绕当前检索结果继续提问。
- 数据库模式：优先连接 MySQL；如果 MySQL 不可用，会自动回退到本地 SQLite 文件 `data/arxiv.db`。
- 前端界面：使用 Streamlit 实现，包含低饱和度学术风格界面、固定标题栏、侧边栏配置和结果卡片。

## 仓库内容

```text
app_streamlit.py                  Streamlit 前端入口
search_backend.py                 检索算法与数据加载核心
ai_search.py                      DeepSeek 查询改写、RAG 综述和追问逻辑
requirements.txt                  Python 依赖
run_quick_benchmarks.py           快速基准测试脚本
run_end2end_test.py               端到端测试脚本
scripts/setup_env.ps1             Windows 一键环境准备脚本
data/sample_arxiv_20000.jsonl     20000 条 arXiv 样例数据
data/arxiv.db                     SQLite 数据库文件
data/benchmarks_summary.csv       基准测试结果摘要
database/create_database.py       数据库建表脚本
database/import_data.py           JSONL 数据导入数据库脚本
database/db_config.py             MySQL 连接配置，默认从环境变量读取
```

本仓库没有提交 arXiv 原始完整快照文件，只保留课程演示所需的 20000 条样例数据和对应 SQLite 数据库。

## 快速运行

推荐使用 Python 3.8 或更高版本。Windows PowerShell 下可以按下面步骤运行：

```powershell
cd ArXiv-Smart-Search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_streamlit.py --server.port 8502
```

启动后在浏览器打开：

```text
http://localhost:8502
```

如果不想手动配置环境，也可以运行脚本：

```powershell
powershell -ExecutionPolicy ByPass -File "scripts/setup_env.ps1"
streamlit run app_streamlit.py --server.port 8502
```

## 数据说明

默认数据文件为：

```text
data/sample_arxiv_20000.jsonl
```

该文件每行是一篇论文的 JSON 对象，主要字段包括：

```text
id, title, abstract, authors, categories, update_date, versions, authors_parsed
```

项目还包含 SQLite 数据库：

```text
data/arxiv.db
```

其中 `papers` 表包含同样的 20000 条论文记录。前端默认读取 JSONL；如果在侧边栏勾选“使用 MySQL 数据库”，系统会先尝试连接 MySQL，失败后自动读取 `data/arxiv.db`。

## 数据库配置

MySQL 是可选项。需要连接 MySQL 时，可以通过环境变量配置：

```powershell
$env:ARXIV_DB_HOST="localhost"
$env:ARXIV_DB_USER="root"
$env:ARXIV_DB_PASSWORD="your_password"
$env:ARXIV_DB_NAME="arxiv_db"
$env:ARXIV_DB_CHARSET="utf8mb4"
```

如果没有 MySQL 或配置失败，系统会自动回退到 SQLite，不影响演示运行。

如需重新生成数据库，可以运行：

```powershell
python database/create_database.py
python database/import_data.py --input data/sample_arxiv_20000.jsonl --limit 20000 --batch 500 --to auto
```

## 增强检索配置

增强检索功能需要 DeepSeek API Key。可以在页面侧边栏直接填写，也可以通过环境变量配置：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"
```

如果没有配置 API Key，传统检索功能仍然可以正常使用。

## 检索算法说明

- TF-IDF：使用 `sklearn.feature_extraction.text.TfidfVectorizer`，通过余弦相似度计算查询和论文文本之间的相关性。
- BM25：使用 `rank_bm25.BM25Okapi`，适合关键词检索和短查询。
- Sentence-BERT：使用 `sentence-transformers/all-MiniLM-L6-v2` 生成语义向量，适合语义相近但词面不同的查询。
- FAISS：后端代码中保留了可选的 FAISS HNSW 接口，用于后续扩展大规模向量检索。

首次使用 Sentence-BERT 时需要联网下载模型，速度会比 TF-IDF 和 BM25 慢。课堂演示时建议优先使用 TF-IDF 或 BM25，确认环境稳定后再测试 Sentence-BERT。

## 基准测试结果

快速基准测试结果保存在：

```text
data/benchmarks_summary.csv
```

摘要结果如下：

| 数据规模 | 算法 | 构建时间 | 平均查询时间 |
|---|---:|---:|---:|
| 1k | TF-IDF | 0.485s | 0.0021s |
| 1k | BM25 | 0.089s | 0.0058s |
| 1k | SBERT | 60.58s | 0.0269s |
| 10k | TF-IDF | 8.37s | 0.0426s |
| 10k | BM25 | 1.87s | 0.1396s |
| 10k | SBERT | 377.79s | 0.0285s |

## 常见问题

### 页面启动后没有结果

先确认依赖已经安装，并且在项目根目录运行：

```powershell
pip install -r requirements.txt
streamlit run app_streamlit.py --server.port 8502
```

### 勾选数据库后报 MySQL 错误

正常情况下不会中断运行。系统会先尝试 MySQL，失败后回退到：

```text
data/arxiv.db
```

如果仍然报错，请确认 `data/arxiv.db` 文件存在。

### Sentence-BERT 很慢

这是正常现象。首次运行需要下载模型并计算嵌入。演示时可以先使用 TF-IDF 或 BM25。

### DeepSeek 不可用

检查 API Key 是否填写正确，或先关闭增强检索功能。传统检索不依赖 API Key。
