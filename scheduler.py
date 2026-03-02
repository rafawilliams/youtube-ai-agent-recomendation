"""
Scheduler para ejecutar el pipeline de YouTube AI Agent automáticamente.

Uso:
    python scheduler.py          → inicia el scheduler (bloquea la terminal)
    python scheduler.py --now    → ejecuta el pipeline inmediatamente y sale

Configuración en .env:
    SCHEDULE_HOUR=15             → hora de ejecución (formato 24h, por defecto 15)
    SCHEDULE_MINUTE=0            → minuto de ejecución (por defecto 0)
    SCHEDULE_TIMEZONE=America/Panama  → timezone (por defecto America/Panama)
"""
import sys
import os
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

# ── Cargar configuración ─────────────────────────────────────────────────────
load_dotenv()

SCHEDULE_HOUR     = int(os.getenv('SCHEDULE_HOUR', '15'))
SCHEDULE_MINUTE   = int(os.getenv('SCHEDULE_MINUTE', '0'))
SCHEDULE_TIMEZONE = pytz.timezone(os.getenv('SCHEDULE_TIMEZONE', 'America/Panama'))

# ── Agregar src al path (necesario para los imports de main.py) ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Logging (centralizado en src/logger.py) ──────────────────────────────────
from logger import setup_logging
setup_logging()
log = logging.getLogger('scheduler')


# ── Tarea programada ─────────────────────────────────────────────────────────

def run_pipeline():
    """Ejecuta el pipeline completo de extracción, análisis y recomendación."""
    start = datetime.now(SCHEDULE_TIMEZONE)
    log.info("=" * 60)
    log.info("INICIO DE EJECUCIÓN AUTOMÁTICA")
    log.info(f"Hora: {start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    log.info("=" * 60)

    try:
        import main as pipeline
        pipeline.main()

        elapsed = (datetime.now(SCHEDULE_TIMEZONE) - start).seconds // 60
        log.info(f"✓ Pipeline completado en ~{elapsed} min")

    except Exception as e:
        log.error(f"❌ Error en la ejecución: {e}", exc_info=True)


# ── Punto de entrada ─────────────────────────────────────────────────────────

def main():
    # Modo --now: ejecuta una vez y sale
    if '--now' in sys.argv:
        log.info("Modo --now: ejecutando pipeline inmediatamente...")
        run_pipeline()
        return

    # Modo normal: scheduler bloqueante
    scheduler = BlockingScheduler(timezone=SCHEDULE_TIMEZONE)

    trigger = CronTrigger(
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        timezone=SCHEDULE_TIMEZONE,
    )

    scheduler.add_job(
        run_pipeline,
        trigger=trigger,
        id='daily_pipeline',
        name='YouTube AI Agent — pipeline diario',
        misfire_grace_time=3600,   # tolerar hasta 1 hora de retraso si el PC estaba apagado
        coalesce=True,             # si se perdieron varias ejecuciones, correr solo una
    )

    hora_fmt = f"{SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d}"
    tz_name  = SCHEDULE_TIMEZONE.zone

    log.info("=" * 60)
    log.info("  YouTube AI Agent — Scheduler iniciado")
    log.info(f"  Ejecución diaria a las {hora_fmt} ({tz_name})")
    log.info("  Presiona Ctrl+C para detener")
    log.info("=" * 60)

    # Mostrar próxima ejecución (calculada manualmente para compatibilidad con APScheduler 3.x y 4.x)
    now = datetime.now(SCHEDULE_TIMEZONE)
    next_run = now.replace(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE, second=0, microsecond=0)
    if next_run <= now:
        # Ya pasó la hora de hoy — la próxima es mañana
        next_run = next_run + timedelta(days=1)
    log.info(f"Próxima ejecución: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler detenido por el usuario.")
        scheduler.shutdown(wait=False)


if __name__ == '__main__':
    main()
