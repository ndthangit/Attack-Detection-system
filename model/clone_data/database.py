from psycopg2 import pool
from dotenv import load_dotenv
import os

load_dotenv()

connection_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
)

def get_db_from_pool():
    """
    Lấy một kết nối từ connection pool.
    """
    return connection_pool.getconn()

def release_db_to_pool(conn):
    """
    Trả kết nối về connection pool.
    """
    connection_pool.putconn(conn)