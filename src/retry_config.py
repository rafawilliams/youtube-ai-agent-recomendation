"""
Configuraciones reutilizables de retry con backoff exponencial.

Cada configuración define qué excepciones son transitorias (retriables),
cuántos intentos hacer, y el backoff entre intentos.

Uso:
    from retry_config import retry_google_api

    @retry_google_api
    def mi_llamada_a_youtube():
        ...
"""
import logging

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

log = logging.getLogger(__name__)


# ── Helpers para determinar si una excepción es transitoria ─────────────


def _is_transient_google_api_error(exc: BaseException) -> bool:
    """True si es un HttpError de Google con status transitorio (429, 5xx)."""
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        return False
    if not isinstance(exc, HttpError):
        return False
    status = exc.resp.status
    return status in (429, 500, 502, 503, 504)


def _is_transient_anthropic_error(exc: BaseException) -> bool:
    """True si es un error transitorio de la API de Anthropic."""
    try:
        from anthropic import (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
        )
    except ImportError:
        return False
    # Errores de red y timeout siempre son transitorios
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    # Errores de servidor (5xx, 529 overloaded) son transitorios
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


def _is_transient_http_error(exc: BaseException) -> bool:
    """True si es un error transitorio de requests (conexión, timeout, 429/5xx)."""
    import requests
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status in (429, 500, 502, 503, 504)
    return False


def _is_transient_db_error(exc: BaseException) -> bool:
    """True si es un error de conexión transitorio de PyMySQL."""
    try:
        from pymysql.err import OperationalError
    except ImportError:
        return False
    if not isinstance(exc, OperationalError):
        return False
    # errno 2003 = "Can't connect to MySQL server"
    # errno 2006 = "MySQL server has gone away"
    # errno 2013 = "Lost connection to MySQL server"
    errno = exc.args[0] if exc.args else 0
    return errno in (2003, 2006, 2013)


# ── Configuraciones reutilizables de retry ──────────────────────────────
# Todas: 3 intentos, backoff exponencial (2s → 4s → 8s, max 10s),
#         log WARNING antes de cada retry, reraise al agotar intentos.

# Google APIs (YouTube Data, YouTube Analytics, Calendar, Sheets)
retry_google_api = retry(
    retry=retry_if_exception(_is_transient_google_api_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)

# Claude / Anthropic API
retry_anthropic = retry(
    retry=retry_if_exception(_is_transient_anthropic_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)

# HTTP requests (Telegram)
retry_http = retry(
    retry=retry_if_exception(_is_transient_http_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)

# Base de datos (solo conexión)
retry_database = retry(
    retry=retry_if_exception(_is_transient_db_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
