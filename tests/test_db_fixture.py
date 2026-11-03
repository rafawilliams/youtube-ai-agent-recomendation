"""
Fixture `test_db`: MariaDB local de pruebas (constitución §22).

Los tests marcados `db` usan DB_NAME_TEST (por defecto `youtube_test`), nunca
la BD de producción, y se saltan si MariaDB no está disponible.
"""

import pytest
from db_support import db_reachable, resolve_test_db_name


class TestResolveTestDbName:
    def test_defaults_to_youtube_test(self):
        assert resolve_test_db_name({"DB_NAME": "youtube_analytics"}) == "youtube_test"

    def test_uses_db_name_test_when_set(self):
        env = {"DB_NAME": "youtube_analytics", "DB_NAME_TEST": "otra_test"}
        assert resolve_test_db_name(env) == "otra_test"

    def test_refuses_production_database(self):
        env = {"DB_NAME": "youtube_analytics", "DB_NAME_TEST": "youtube_analytics"}
        with pytest.raises(ValueError, match="producción"):
            resolve_test_db_name(env)

    def test_refuses_default_production_name(self):
        # Sin DB_NAME, la producción es el valor por defecto de src/database.py
        with pytest.raises(ValueError, match="producción"):
            resolve_test_db_name({"DB_NAME_TEST": "youtube_analytics"})


class TestDbReachable:
    def test_unreachable_server_returns_false_quickly(self):
        # Puerto cerrado: debe responder False sin los reintentos de producción
        assert db_reachable(host="127.0.0.1", port=1, user="x", password="x") is False


@pytest.mark.db
class TestTestDbFixture:
    def test_connected_to_test_database(self, test_db):
        cur = test_db.conn.cursor()
        cur.execute("SELECT DATABASE() AS db")
        assert cur.fetchone()["db"] == resolve_test_db_name()

    def test_tables_created_by_production_schema(self, test_db):
        cur = test_db.conn.cursor()
        cur.execute("SHOW TABLES LIKE 'channels'")
        assert cur.fetchone() is not None

    def test_writes_are_isolated_between_tests_first(self, test_db):
        _insert_channel(test_db, "UC_FIXTURE")
        assert _channel_count(test_db) == 1

    def test_writes_are_isolated_between_tests_second(self, test_db):
        # Si la fixture no vaciara las tablas, aquí habría 1 fila del test anterior
        assert _channel_count(test_db) == 0
        _insert_channel(test_db, "UC_FIXTURE")
        assert _channel_count(test_db) == 1


def _insert_channel(db, channel_id):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO channels (channel_id, channel_name) VALUES (%s, %s)",
        (channel_id, "Canal de prueba"),
    )
    db.conn.commit()


def _channel_count(db):
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM channels")
    return cur.fetchone()["n"]
