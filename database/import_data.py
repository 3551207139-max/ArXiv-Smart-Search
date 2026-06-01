# database/import_data.py
import argparse
import json
import os
import sqlite3

try:
    from database.clean_text import clean_text
except Exception:
    from clean_text import clean_text

try:
    import pymysql
    from database.db_config import DB_CONFIG
    HAS_PYMYSQL = True
except Exception:
    HAS_PYMYSQL = False


def create_table(conn, cursor, use_mysql: bool):
    if use_mysql:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            id VARCHAR(50) PRIMARY KEY,
            submitter TEXT,
            authors TEXT,
            title TEXT,
            comments TEXT,
            journal_ref TEXT,
            doi VARCHAR(255),
            report_no TEXT,
            categories VARCHAR(255),
            license TEXT,
            abstract LONGTEXT,
            versions JSON,
            update_date DATE,
            authors_parsed JSON
        )
        """
        cursor.execute(create_table_sql)
        try:
            cursor.execute("ALTER TABLE papers ADD FULLTEXT(title, abstract)")
        except Exception:
            pass
    else:
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            submitter TEXT,
            authors TEXT,
            title TEXT,
            comments TEXT,
            journal_ref TEXT,
            doi TEXT,
            report_no TEXT,
            categories TEXT,
            license TEXT,
            abstract TEXT,
            versions TEXT,
            update_date TEXT,
            authors_parsed TEXT
        )
        """
        cursor.execute(create_table_sql)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/sample_arxiv_20000.jsonl',
                   help='JSONL 输入文件路径（每行一个 JSON）')
    p.add_argument('--limit', '-n', type=int, default=0, help='最多导入多少条，0 为全部')
    p.add_argument('--batch', '-b', type=int, default=500)
    p.add_argument('--to', choices=['auto', 'mysql', 'sqlite', 'both'], default='auto',
                   help='目标数据库：auto=优先 MySQL（若可用），否则 SQLite；both=同时写入两者')
    return p.parse_args()


def open_db(use_mysql: bool):
    """打开并返回 (conn, cursor, is_mysql)"""
    if use_mysql and HAS_PYMYSQL:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset'],
            autocommit=False
        )
        cursor = conn.cursor()
        return conn, cursor, True
    else:
        db_path = 'data/arxiv.db'
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        return conn, cursor, False


def import_from_jsonl(infile: str, conn, cursor, use_mysql: bool, limit: int = 0, batch_size: int = 500):
    """从 JSONL 导入到已打开的数据库连接（不负责打开/关闭连接）"""
    placeholders = '(?,?,?,?,?,?,?,?,?,?,?,?,?,?)' if not use_mysql else '(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
    if use_mysql:
        insert_sql = '''INSERT IGNORE INTO papers (
            id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
        ) VALUES ''' + placeholders
    else:
        insert_sql = '''INSERT OR IGNORE INTO papers (
            id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
        ) VALUES ''' + placeholders

    count = 0
    batch = []

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if limit and count >= limit:
                break
            try:
                paper = json.loads(line)
            except Exception:
                continue

            count += 1

            paper_id = paper.get('id')
            submitter = paper.get('submitter')
            authors = paper.get('authors')
            title = paper.get('title') or ''
            comments = paper.get('comments')
            journal_ref = paper.get('journal-ref')
            doi = paper.get('doi')
            report_no = paper.get('report-no')
            categories = paper.get('categories')
            license_info = paper.get('license')

            # Prefer cleaned combined field
            combined = paper.get('title_abstract')
            if not combined:
                combined = f"{title}\n{paper.get('abstract','')}"
            cleaned = clean_text(combined)

            versions = json.dumps(paper.get('versions', []), ensure_ascii=False)
            authors_parsed = json.dumps(paper.get('authors_parsed', []), ensure_ascii=False)

            update_date = paper.get('update_date')

            batch.append((
                paper_id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license_info, cleaned, versions, update_date, authors_parsed
            ))

            if len(batch) >= batch_size:
                try:
                    cursor.executemany(insert_sql, batch)
                    conn.commit()
                except Exception as e:
                    print('批量插入出错，尝试逐条插入，错误：', e)
                    for row in batch:
                        try:
                            cursor.execute(insert_sql, row)
                        except Exception as e2:
                            print('单条插入失败 id=', row[0], e2)
                    conn.commit()
                print(f'已导入 {count} 条')
                batch = []

    # flush remaining
    if batch:
        try:
            cursor.executemany(insert_sql, batch)
            conn.commit()
        except Exception as e:
            print('插入剩余数据出错：', e)

    return count


def main():

    args = parse_args()

    infile = args.input
    limit = args.limit
    batch_size = args.batch
    to = args.to

    # Decide targets
    target_mysql = False
    target_sqlite = False
    if to == 'auto':
        target_mysql = HAS_PYMYSQL
        target_sqlite = not HAS_PYMYSQL
    elif to == 'mysql':
        target_mysql = True
    elif to == 'sqlite':
        target_sqlite = True
    elif to == 'both':
        target_mysql = True
        target_sqlite = True

    # Prepare connections
    mysql_conn = mysql_cursor = None
    sqlite_conn = sqlite_cursor = None
    if target_mysql and not HAS_PYMYSQL:
        print('目标包含 MySQL，但环境中未检测到 pymysql，MySQL 导入将被跳过')
        target_mysql = False

    if target_mysql:
        mysql_conn, mysql_cursor, _ = open_db(use_mysql=True)
        create_table(mysql_conn, mysql_cursor, True)
        mysql_conn.commit()

    if target_sqlite:
        sqlite_conn, sqlite_cursor, _ = open_db(use_mysql=False)
        create_table(sqlite_conn, sqlite_cursor, False)
        sqlite_conn.commit()

    total_count = 0

    # If writing to both, we will perform two separate insert operations
    if target_mysql and target_sqlite:
        # Import once and write to both by reading JSONL and inserting into each
        total_count = 0
        with open(infile, 'r', encoding='utf-8') as f:
            batch_mysql = []
            batch_sqlite = []
            for line in f:
                if not line.strip():
                    continue
                if limit and total_count >= limit:
                    break
                try:
                    paper = json.loads(line)
                except Exception:
                    continue
                total_count += 1
                # prepare row same as import_from_jsonl
                paper_id = paper.get('id')
                submitter = paper.get('submitter')
                authors = paper.get('authors')
                title = paper.get('title') or ''
                comments = paper.get('comments')
                journal_ref = paper.get('journal-ref')
                doi = paper.get('doi')
                report_no = paper.get('report-no')
                categories = paper.get('categories')
                license_info = paper.get('license')
                combined = paper.get('title_abstract')
                if not combined:
                    combined = f"{title}\n{paper.get('abstract','')}"
                cleaned = clean_text(combined)
                versions = json.dumps(paper.get('versions', []), ensure_ascii=False)
                authors_parsed = json.dumps(paper.get('authors_parsed', []), ensure_ascii=False)
                update_date = paper.get('update_date')

                row = (
                    paper_id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license_info, cleaned, versions, update_date, authors_parsed
                )

                batch_mysql.append(row)
                batch_sqlite.append(row)

                if len(batch_mysql) >= batch_size:
                    try:
                        mysql_cursor.executemany('''INSERT IGNORE INTO papers (
                            id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ''', batch_mysql)
                        mysql_conn.commit()
                    except Exception as e:
                        print('MySQL 批量插入错误：', e)
                    try:
                        sqlite_cursor.executemany('''INSERT OR IGNORE INTO papers (
                            id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ''', batch_sqlite)
                        sqlite_conn.commit()
                    except Exception as e:
                        print('SQLite 批量插入错误：', e)
                    print(f'已导入 {total_count} 条')
                    batch_mysql = []
                    batch_sqlite = []

        # flush
        if batch_mysql:
            try:
                mysql_cursor.executemany('''INSERT IGNORE INTO papers (
                    id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ''', batch_mysql)
                mysql_conn.commit()
            except Exception as e:
                print('MySQL 插入剩余批次错误：', e)
            try:
                sqlite_cursor.executemany('''INSERT OR IGNORE INTO papers (
                    id, submitter, authors, title, comments, journal_ref, doi, report_no, categories, license, abstract, versions, update_date, authors_parsed
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', batch_sqlite)
                sqlite_conn.commit()
            except Exception as e:
                print('SQLite 插入剩余批次错误：', e)

    else:
        # Single target import via helper
        if target_mysql:
            total_count = import_from_jsonl(infile, mysql_conn, mysql_cursor, True, limit=limit, batch_size=batch_size)
        elif target_sqlite:
            total_count = import_from_jsonl(infile, sqlite_conn, sqlite_cursor, False, limit=limit, batch_size=batch_size)

    # Close connections
    if target_mysql:
        try:
            mysql_cursor.close()
            mysql_conn.close()
        except Exception:
            pass
    if target_sqlite:
        try:
            sqlite_cursor.close()
            sqlite_conn.close()
        except Exception:
            pass

    print('导入完成，总计处理条数：', total_count)


if __name__ == '__main__':
    main()


