"""Soporte de la BD local de pruebas (constitución §22).

Los tests de integración marcados `db` usan una MariaDB local aparte, nunca la
de producción. Este módulo decide su nombre y comprueba que el servidor responde.
"""

import os

import pymysql

DEFAULT_TEST_DB = "youtube_test"
# Mismo valor por defecto que src/database.py para DB_NAME
DEFAULT_PRODUCTION_DB = "youtube_analytics"


def resolve_test_db_name(env=None) -> str:
    """Nombre de la BD de pruebas. Lanza ValueError si coincide con producción."""
    env = os.environ if env is None else env
    production = env.get("DB_NAME", DEFAULT_PRODUCTION_DB)
    test_name = env.get("DB_NAME_TEST", DEFAULT_TEST_DB)
    if test_name in (production, DEFAULT_PRODUCTION_DB):
        raise ValueError(
            f"DB_NAME_TEST='{test_name}' es la BD de producción: los tests nunca la usan"
        )
    return test_name


def db_reachable(host, port, user, password) -> bool:
    """True si MariaDB acepta la conexión. Un intento corto, sin reintentos."""
    try:
        conn = pymysql.connect(
            host=host, port=int(port), user=user, password=password, connect_timeout=2
        )
    except pymysql.MySQLError:
        return False
    conn.close()
    return True
