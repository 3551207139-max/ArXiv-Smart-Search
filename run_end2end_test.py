"""
End-to-end quick verification script.
- Attempts to load up to 20000 docs from MySQL; if fails, falls back to JSONL.
- Builds TF-IDF and BM25 on full set and runs a few sample queries.
- Builds SBERT on a 1k subset to verify SBERT search works.
"""
import time
from search_backend import SearchEngine, load_jsonl, load_from_mysql, make_title_abstract_field

MAX_DOCS = 20000
SAMPLE_QUERIES = [
    'deep learning',
    'graph neural networks',
    'transformer architecture',
]


def load_docs():
    try:
        docs = load_from_mysql(max_docs=MAX_DOCS)
        print('Loaded from MySQL, docs=', len(docs))
        return docs
    except Exception as e:
        print('Load from MySQL failed, falling back to JSONL:', e)
        docs = load_jsonl('data/sample_arxiv_20000.jsonl', max_docs=MAX_DOCS)
        print('Loaded from JSONL, docs=', len(docs))
        return docs


def main():
    docs = load_docs()
    docs = make_title_abstract_field(docs)

    print('\n-- TF-IDF & BM25 test --')
    engine = SearchEngine(docs)
    t0 = time.time()
    engine.build_tfidf()
    print('TF-IDF built in', time.time()-t0)

    t0 = time.time()
    engine.build_bm25()
    print('BM25 built in', time.time()-t0)

    for q in SAMPLE_QUERIES:
        print('\nQuery:', q)
        r_tfidf = engine.search_tfidf(q, top_k=3)
        print('TF-IDF top:', [(r[0], r[2].get('title','')[:80]) for r in r_tfidf])
        r_bm25 = engine.search_bm25(q, top_k=3)
        print('BM25   top:', [(r[0], r[2].get('title','')[:80]) for r in r_bm25])

    # SBERT small test on 1k subset
    subset = docs[:1000]
    print('\n-- SBERT small subset test (1k) --')
    engine_s = SearchEngine(subset)
    t0 = time.time()
    engine_s.build_sbert()
    print('SBERT built on 1k in', time.time()-t0)

    for q in SAMPLE_QUERIES:
        r = engine_s.search_sbert(q, top_k=3)
        print('SBERT top for', q, ':', [(r0[0], r0[2].get('title','')[:80]) for r0 in r])

if __name__ == '__main__':
    main()
