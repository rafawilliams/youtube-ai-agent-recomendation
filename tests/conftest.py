"""
Fixtures compartidos para la suite de tests del YouTube AI Agent.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_videos_df():
    """DataFrame con 20 videos variados (mix de Shorts y videos largos)."""
    np.random.seed(42)
    base_date = pd.Timestamp("2025-12-01", tz="UTC")

    records = []
    titles = [
        "Tutorial de Python para principiantes",
        "Reacción al nuevo trailer de Marvel",
        "Top 10 trucos de productividad",
        "Mi rutina de mañana - Vlog diario",
        "Review del iPhone 16 Pro Max",
        "Cómo hacer pan casero paso a paso",
        "Las últimas noticias de tecnología",
        "Challenge: 24 horas sin internet",
        "5 mejores apps para estudiantes",
        "Unboxing de la PS6 edición especial",
        "Sketch de comedia: el jefe malvado",
        "Aprende JavaScript en 10 minutos",
        "Reseña completa del Tesla Model Y",
        "Tips para mejorar tu canal de YouTube",
        "Reto de cocina con los ojos vendados",
        "Prank a mi hermano (no salió bien)",
        "Guía completa de edición de video",
        "Día en mi vida como programador",
        "Los peores errores al empezar en YouTube",
        "React vs Vue: cuál elegir en 2026?",
    ]
    descriptions = [
        "En este video aprenderás los fundamentos de Python desde cero.",
        "Reaccionamos al trailer más esperado del año.",
        "Descubre los mejores trucos para ser más productivo.",
        "Acompáñame en un día normal de mi vida.",
        "Análisis detallado del nuevo iPhone.",
        "Receta fácil de pan casero para principiantes.",
        "Resumen de las noticias más importantes de la semana.",
        "Intentamos sobrevivir sin internet por un día completo.",
        "Las aplicaciones que todo estudiante necesita.",
        "Abrimos la nueva consola de Sony.",
        "Sketch de humor sobre la vida en oficina.",
        "Curso express de JavaScript para empezar hoy.",
        "Todo lo que necesitas saber sobre este auto eléctrico.",
        "Consejos prácticos para crecer en YouTube.",
        "Un reto de cocina que salió completamente mal.",
        "Le hice una broma épica a mi hermano.",
        "De principiante a pro en edición de video.",
        "Cómo es trabajar como programador remoto.",
        "Evita estos errores comunes de YouTubers novatos.",
        "Comparativa técnica de los dos frameworks más populares.",
    ]
    tag_options = [
        "python,programación,tutorial,código",
        "marvel,reacción,trailer,cine",
        "productividad,tips,top10,hacks",
        "vlog,rutina,mañana,lifestyle",
        "iphone,review,apple,tecnología",
        "cocina,receta,pan,casero",
        "noticias,tech,actualización",
        "challenge,internet,reto,24h",
        "apps,estudiantes,top5,educación",
        "unboxing,ps6,sony,gaming",
        "comedia,sketch,humor,oficina",
        "javascript,programación,web,curso",
        "tesla,review,auto,eléctrico",
        "youtube,tips,canal,crecer",
        "cocina,reto,challenge,comida",
        "prank,broma,hermano,humor",
        "edición,video,guía,adobe",
        "vlog,programador,vida,remoto",
        "youtube,errores,principiante,tips",
        "react,vue,javascript,framework",
    ]

    for i in range(20):
        is_short = i % 3 == 0  # ~7 shorts, ~13 long
        duration_secs = (
            np.random.randint(15, 60) if is_short else np.random.randint(300, 1800)
        )
        views = np.random.randint(500, 50000)
        likes = int(views * np.random.uniform(0.02, 0.08))
        comments = int(views * np.random.uniform(0.005, 0.02))
        engagement = round((likes + comments) / max(views, 1) * 100, 2)

        records.append(
            {
                "video_id": f"vid_{i:03d}",
                "title": titles[i],
                "description": descriptions[i],
                "published_at": base_date
                + pd.Timedelta(days=i * 2, hours=np.random.randint(8, 22)),
                "channel_id": "UC_test_channel_001",
                "channel_title": "Canal de Prueba",
                "duration": f"PT{duration_secs}S",
                "duration_seconds": duration_secs,
                "is_short": is_short,
                "video_type": "Short" if is_short else "Video Largo",
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "tags": tag_options[i],
                "category_id": "22",
                "privacy_status": "public",
                "engagement_rate": engagement,
                "days_since_published": (40 - i * 2),
                "content_category": "other",
            }
        )

    return pd.DataFrame(records)


@pytest.fixture
def sample_stats():
    """Dict que simula la salida de AIAnalyzer._calculate_statistics()."""
    return {
        "total_videos": 20,
        "total_shorts": 7,
        "total_long_videos": 13,
        "videos_last_30_days": 10,
        "avg_views": 15000.0,
        "median_views": 12000.0,
        "total_views": 300000,
        "avg_engagement_rate": 5.5,
        "shorts_avg_views": 20000.0,
        "long_videos_avg_views": 12000.0,
        "shorts_avg_engagement": 6.5,
        "long_videos_avg_engagement": 4.8,
        "recent_avg_views": 18000.0,
        "recent_shorts_count": 3,
        "recent_long_videos_count": 7,
        "best_video_views": 48000,
        "best_video_title": "Top 10 trucos de productividad",
        "best_video_type": "Video Largo",
    }


@pytest.fixture
def mock_claude_response():
    """Respuesta realista de Claude con TITULO 1/2/3 + ANALISIS."""
    return """
1. FORMATO RECOMENDADO: Short (9:16)

Un Short tiene el mejor potencial de alcance para este canal dado que los Shorts
promedian 20,000 vistas vs 12,000 de videos largos.

2. TEMA/TIPO DE CONTENIDO: Tutorial rápido de programación

Basado en el rendimiento de "Tutorial de Python para principiantes" que obtuvo
excelentes métricas, recomiendo un tutorial corto sobre un tema trending.

3. RAZÓN: Los tutoriales cortos combinan el formato de mayor alcance (Short) con
la temática de mejor rendimiento del canal.

4. PREDICCIÓN: Espero que este video obtenga entre 18,000 y 25,000 vistas,
superando el promedio del canal.

5. HORARIO: Publicar el miércoles a las 18:00 (hora de Panamá), que es históricamente
el mejor momento.

6. SUGERENCIAS DE TÍTULO:

TÍTULO 1: "Este truco de Python te ahorrará HORAS de trabajo"
ANÁLISIS: Usa curiosidad + beneficio directo. La palabra "truco" genera clicks y
"HORAS" en mayúsculas enfatiza el valor. Ideal para programadores que buscan eficiencia.

TÍTULO 2: "Python en 60 segundos: automatiza tu día"
ANÁLISIS: El formato "en 60 segundos" es perfecto para Shorts y establece expectativas
claras. "Automatiza" es un keyword con alto volumen de búsqueda.

TÍTULO 3: "Lo que NADIE te dice sobre Python (y deberías saber)"
ANÁLISIS: Apela a la curiosidad con "nadie te dice" y crea FOMO. El paréntesis añade
urgencia. Funciona especialmente bien con audiencias intermedias.
"""


@pytest.fixture
def mock_anthropic():
    """Fixture que mockea anthropic.Anthropic para evitar llamadas reales."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Configurar respuesta por defecto
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Respuesta de prueba")]
        mock_client.messages.create.return_value = mock_message

        yield mock_client


@pytest.fixture
def training_videos_df():
    """DataFrame con 25 videos con suficiente varianza para entrenar modelos ML.
    Incluye avg_view_percentage para RetentionPredictor."""
    np.random.seed(123)
    base_date = pd.Timestamp("2025-06-01", tz="UTC")

    records = []
    for i in range(25):
        is_short = i % 4 == 0  # ~6 shorts
        duration_secs = (
            np.random.randint(15, 55) if is_short else np.random.randint(300, 2400)
        )
        # Views con varianza significativa para que el modelo pueda aprender
        views = int(np.random.lognormal(mean=9, sigma=1.2))
        likes = int(views * np.random.uniform(0.02, 0.10))
        comments = int(views * np.random.uniform(0.003, 0.025))
        engagement = round((likes + comments) / max(views, 1) * 100, 2)
        tags_count = np.random.randint(3, 15)
        tags = ",".join([f"tag{j}" for j in range(tags_count)])

        has_number = 1 if np.random.random() > 0.5 else 0
        has_question = 1 if np.random.random() > 0.7 else 0
        title_base = f"Video de prueba {'#' + str(i) if has_number else ''} {'?' if has_question else ''} - Tema {i % 5}"

        records.append(
            {
                "video_id": f"train_{i:03d}",
                "title": title_base,
                "description": f"Descripción del video {i} con contenido variado " * 3,
                "published_at": base_date
                + pd.Timedelta(days=i * 3, hours=np.random.randint(6, 23)),
                "channel_id": "UC_train_channel_001",
                "channel_title": "Canal Entrenamiento",
                "duration": f"PT{duration_secs}S",
                "duration_seconds": duration_secs,
                "is_short": is_short,
                "video_type": "Short" if is_short else "Video Largo",
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "tags": tags,
                "category_id": "22",
                "privacy_status": "public",
                "engagement_rate": engagement,
                "days_since_published": 75 - i * 3,
                "content_category": "tutorial" if i % 3 == 0 else "entertainment",
                "avg_view_percentage": round(np.random.uniform(20, 80), 1),
            }
        )

    return pd.DataFrame(records)


# ── BD local de pruebas (constitución §22) ───────────────────────────


@pytest.fixture(scope="session")
def _test_db_session():
    """Conexión de sesión a DB_NAME_TEST con el esquema real de src/database.py.

    Hace skip si MariaDB no responde. Nunca usa la BD de producción.
    """
    import os

    import pymysql
    from db_support import db_reachable, resolve_test_db_name
    from dotenv import load_dotenv

    load_dotenv()
    test_name = resolve_test_db_name()
    creds = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    if not db_reachable(**creds):
        pytest.skip("MariaDB local no disponible: se saltan los tests marcados `db`")

    admin = pymysql.connect(**creds)
    try:
        admin.cursor().execute(
            f"CREATE DATABASE IF NOT EXISTS `{test_name}` CHARACTER SET utf8mb4"
        )
    finally:
        admin.close()

    previous = os.environ.get("DB_NAME")
    os.environ["DB_NAME"] = test_name
    try:
        from database import YouTubeDatabase

        db = YouTubeDatabase()
    finally:
        if previous is None:
            os.environ.pop("DB_NAME", None)
        else:
            os.environ["DB_NAME"] = previous

    cur = db.conn.cursor()
    cur.execute("SELECT DATABASE() AS db")
    connected = cur.fetchone()["db"]
    if connected != test_name:
        db.conn.close()
        pytest.fail(f"La fixture se conectó a '{connected}' en vez de '{test_name}'")

    yield db
    db.conn.close()


@pytest.fixture
def test_db(_test_db_session):
    """YouTubeDatabase sobre DB_NAME_TEST con todas las tablas vacías."""
    db = _test_db_session
    cur = db.conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [next(iter(row.values())) for row in cur.fetchall()]
    cur.execute("SET FOREIGN_KEY_CHECKS = 0")
    for table in tables:
        cur.execute(f"TRUNCATE TABLE `{table}`")
    cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    db.conn.commit()
    return db
