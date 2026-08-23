"""
Tiny SQLite-backed "orders" database standing in for a real service.

In a real deployment this module would be an async client for your actual
order-management service (a REST call, a gRPC call, or a proper async DB
driver like asyncpg). It's kept as plain sqlite3 here so the whole project
runs with zero external services.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

_SEED_ORDERS = [
    # order_id, user_id, status, amount, item
    ("ORD-1001", "cust-1", "shipped", 42.50, "Wireless Mouse"),
    ("ORD-1002", "cust-1", "delivered", 129.00, "Mechanical Keyboard"),
    ("ORD-1003", "cust-2", "pending", 19.99, "USB-C Cable"),
    ("ORD-1004", "cust-2", "delivered", 89.00, "Webcam 1080p")
]


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    Path(settings.app_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.app_db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_connection():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and seed demo data. Called once at app startup."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                amount REAL NOT NULL,
                item TEXT NOT NULL,
                refunded INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        existing = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        if existing == 0:
            conn.executemany(
                "INSERT INTO orders (order_id, user_id, status, amount, item) VALUES (?, ?, ?, ?, ?)",
                _SEED_ORDERS
            )
        conn.commit()

def get_order(order_id: str, user_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ? AND user_id = ?", (order_id, user_id)
        ).fetchone()
        return dict(row) if row else None

def list_orders(user_id: str) -> list[dict]:
    print("user id", user_id)
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM orders WHERE user_id = ?", (user_id,)).fetchall()
        return [dict(r) for r in rows]

def mark_refunded(order_id: str, user_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE orders SET refunded = 1 WHERE order_id = ? AND user_id = ?",
            (order_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0
