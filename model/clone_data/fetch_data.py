import database
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor


def fetch_data_in_batches(batch_size=10000):
    conn = None
    try:
        conn = database.get_db_from_pool()
        cur = conn.cursor()
        offset = 0
        while True:
            cur.execute(f"SELECT * FROM logs_bgl LIMIT {batch_size} OFFSET {offset};")
            batch = cur.fetchall()
            if not batch:
                break
            yield batch  # Trả về từng batch dữ liệu
            offset += batch_size
    finally:
        if conn:
            database.release_db_to_pool(conn)
