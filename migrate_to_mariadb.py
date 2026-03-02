"""
Script de migración de datos SQLite → MariaDB
Ejecutar una sola vez para mover los datos existentes.
"""
import sqlite3
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

SQLITE_PATH = "data/youtube_analytics.db"


def get_mariadb_conn():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'youtube_analytics'),
        charset='utf8mb4',
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor
    )


def migrate_table(sqlite_cur, maria_cur, table: str, insert_sql: str, row_to_tuple):
    sqlite_cur.execute(f"SELECT * FROM {table}")
    rows = sqlite_cur.fetchall()
    if not rows:
        print(f"  {table}: sin datos, omitiendo.")
        return 0

    count = 0
    for row in rows:
        try:
            maria_cur.execute(insert_sql, row_to_tuple(row))
            count += 1
        except pymysql.err.IntegrityError:
            pass  # Duplicado, ignorar

    print(f"  {table}: {count}/{len(rows)} filas migradas.")
    return count


def main():
    if not os.path.exists(SQLITE_PATH):
        print(f"No se encontró la base de datos SQLite en '{SQLITE_PATH}'.")
        print("No hay datos que migrar.")
        return

    print("Conectando a SQLite...")
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    print("Conectando a MariaDB...")
    try:
        maria_conn = get_mariadb_conn()
    except pymysql.err.OperationalError as e:
        print(f"Error conectando a MariaDB: {e}")
        print("\nVerifica que en tu .env estén configuradas las variables:")
        print("  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME")
        return

    maria_cur = maria_conn.cursor()

    print("\nMigrando datos...\n")

    # channels
    migrate_table(
        sqlite_cur, maria_cur,
        table="channels",
        insert_sql="""
            INSERT IGNORE INTO channels
            (channel_id, channel_name, description, subscriber_count,
             video_count, view_count, created_at, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        row_to_tuple=lambda r: (
            r["channel_id"], r["channel_name"], r["description"],
            r["subscriber_count"], r["video_count"], r["view_count"],
            r["created_at"], r["last_updated"]
        )
    )

    # videos
    migrate_table(
        sqlite_cur, maria_cur,
        table="videos",
        insert_sql="""
            INSERT IGNORE INTO videos
            (video_id, channel_id, title, description, published_at,
             duration_seconds, is_short, video_type, tags, category_id, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        row_to_tuple=lambda r: (
            r["video_id"], r["channel_id"], r["title"], r["description"],
            r["published_at"], r["duration_seconds"], r["is_short"],
            r["video_type"], r["tags"], r["category_id"], r["last_updated"]
        )
    )

    # video_metrics
    migrate_table(
        sqlite_cur, maria_cur,
        table="video_metrics",
        insert_sql="""
            INSERT INTO video_metrics
            (video_id, view_count, like_count, comment_count, engagement_rate, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
        row_to_tuple=lambda r: (
            r["video_id"], r["view_count"], r["like_count"],
            r["comment_count"], r["engagement_rate"], r["recorded_at"]
        )
    )

    # recommendations
    migrate_table(
        sqlite_cur, maria_cur,
        table="recommendations",
        insert_sql="""
            INSERT INTO recommendations
            (channel_id, recommendation_date, recommended_type,
             recommended_topic, reasoning, predicted_performance, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        row_to_tuple=lambda r: (
            r["channel_id"], r["recommendation_date"], r["recommended_type"],
            r["recommended_topic"], r["reasoning"], r["predicted_performance"],
            r["created_at"]
        )
    )

    maria_conn.commit()
    sqlite_conn.close()
    maria_conn.close()

    print("\n✓ Migración completada.")


if __name__ == "__main__":
    main()
