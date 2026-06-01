"""
快速基准脚本：在指定文档数量上测量 TF-IDF、BM25、SBERT 的构建与查询时间。
用法示例：
  d:\downloads\...\.venv\Scripts\python.exe run_quick_benchmarks.py --input data/sample_arxiv_20000.jsonl --sizes 1000 10000
"""
import time
import random
import argparse
import numpy as np
from search_backend import load_jsonl, make_title_abstract_field, SearchEngine


def benchmark(docs, n_queries=100):
    engine = SearchEngine(docs)
    results = {}

    # Prepare queries: 从文档中取出若干 title_abstract 片段
    corpus_texts = [d.get('title_abstract','') for d in docs]
    sample_queries = []
    for _ in range(n_queries):
        t = random.choice(corpus_texts)
        if not t:
            t = 'machine learning'
        # 使用标题或前 30 个词作为查询
        q = ' '.join(t.split()[:30])
        sample_queries.append(q)

    # TF-IDF
    t0 = time.time()
    engine.build_tfidf()
    t1 = time.time()
    tfidf_build = t1 - t0
    q_times = []
    for q in sample_queries:
        tq0 = time.time()
        engine.search_tfidf(q, top_k=10)
        tq1 = time.time()
        q_times.append(tq1 - tq0)
    results['tfidf'] = {'build_time': tfidf_build, 'avg_query': float(np.mean(q_times))}

    # BM25
    t0 = time.time()
    engine.build_bm25()
    t1 = time.time()
    bm25_build = t1 - t0
    q_times = []
    for q in sample_queries:
        tq0 = time.time()
        engine.search_bm25(q, top_k=10)
        tq1 = time.time()
        q_times.append(tq1 - tq0)
    results['bm25'] = {'build_time': bm25_build, 'avg_query': float(np.mean(q_times))}

    # SBERT (may be slow on first run)
    t0 = time.time()
    engine.build_sbert()
    t1 = time.time()
    sbert_build = t1 - t0
    q_times = []
    for q in sample_queries:
        tq0 = time.time()
        engine.search_sbert(q, top_k=10)
        tq1 = time.time()
        q_times.append(tq1 - tq0)
    results['sbert'] = {'build_time': sbert_build, 'avg_query': float(np.mean(q_times))}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000,10000])
    parser.add_argument('--n-queries', type=int, default=100)
    args = parser.parse_args()

    all_docs = load_jsonl(args.input, max_docs=None)
    print('total docs available:', len(all_docs))

    for size in args.sizes:
        docs = all_docs[:size]
        docs = make_title_abstract_field(docs)
        print('\n=== Benchmark size:', size, '===')
        start = time.time()
        res = benchmark(docs, n_queries=args.n_queries)
        dur = time.time() - start
        print('results:', res)
        print('total time for this size:', dur)

if __name__ == '__main__':
    main()
