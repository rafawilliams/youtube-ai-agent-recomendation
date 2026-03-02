"""
Configuración centralizada de logging para YouTube AI Agent.

Uso desde cualquier módulo:
    import logging
    log = logging.getLogger(__name__)

setup_logging() se debe llamar UNA sola vez al inicio del proceso
(en main.py o scheduler.py). Las llamadas posteriores son idempotentes.
"""
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_CONFIGURED = False

# Directorio de logs relativo a la raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / 'logs'

# Formato unificado
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s — %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging() -> None:
    """
    Configura el logging global del proyecto.

    - Nivel: controlado por LOG_LEVEL (default INFO).
    - Handlers:
        1. StreamHandler → stdout (consola, igual que antes).
        2. TimedRotatingFileHandler → logs/pipeline.log
           con rotación diaria y retención de 30 archivos.
    - Idempotente: llamar más de una vez no duplica handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)

    # Crear directorio de logs si no existe
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Formatter compartido
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Handler de archivo con rotación diaria
    file_handler = TimedRotatingFileHandler(
        filename=str(_LOG_DIR / 'pipeline.log'),
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8',
    )
    file_handler.suffix = '%Y-%m-%d'
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Configurar root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silenciar loggers verbosos de terceros
    logging.getLogger('apscheduler').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

    _CONFIGURED = True
