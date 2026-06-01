# database/create_database.py
"""Create the `papers` table in MySQL or fall back to SQLite if pymysql missing."""
try:
    import pymysql
    from db_config import DB_CONFIG
    USE_MYSQL = True
except Exception:
    USE_MYSQL = False


if USE_MYSQL:
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            charset=DB_CONFIG["charset"],
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        print(f"数据库 {DB_CONFIG['database']} 已创建/使用 (MySQL)")
    except Exception as e:
        print('无法连接 MySQL，已回退到 SQLite（错误：', e, ')')
        USE_MYSQL = False

if not USE_MYSQL:
    import sqlite3
    import os
    db_path = 'data/arxiv.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    print(f"使用 SQLite，数据库文件: {db_path}")
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
    conn.commit()
    cursor.close()
    conn.close()
    print("数据库初始化完成 (SQLite)")
