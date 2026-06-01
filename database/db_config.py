import os

DB_CONFIG = {
    "host": os.environ.get("ARXIV_DB_HOST", "localhost"),
    "user": os.environ.get("ARXIV_DB_USER", "root"),
    "password": os.environ.get("ARXIV_DB_PASSWORD", ""),
    "database": os.environ.get("ARXIV_DB_NAME", "arxiv_db"),
    "charset": os.environ.get("ARXIV_DB_CHARSET", "utf8mb4"),
}
