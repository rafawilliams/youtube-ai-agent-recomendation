"""
Módulo para gestión de base de datos MariaDB
"""
import json
import logging
import pymysql
import pymysql.cursors
import pandas as pd
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv
from retry_config import retry_database

log = logging.getLogger(__name__)

load_dotenv()


class YouTubeDatabase:
    """Gestiona el almacenamiento de datos de YouTube en MariaDB"""

    def __init__(self):
        """Inicializa la conexión a MariaDB usando variables de entorno"""
        self.conn = self._connect_with_retry()
        self._create_tables()

    @staticmethod
    @retry_database
    def _connect_with_retry():
        """Establece conexión a MariaDB con retry en errores transitorios."""
        return pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'youtube_analytics'),
            charset='utf8mb4',
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _create_tables(self):
        """Crea las tablas necesarias si no existen"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id VARCHAR(255) NOT NULL PRIMARY KEY,
                channel_name VARCHAR(500) NOT NULL,
                description TEXT,
                subscriber_count INTEGER,
                video_count INTEGER,
                view_count INTEGER,
                is_competitor TINYINT(1) NOT NULL DEFAULT 0,
                created_at VARCHAR(50),
                last_updated VARCHAR(50)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR(255) NOT NULL PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                title VARCHAR(1000) NOT NULL,
                description TEXT,
                published_at VARCHAR(50) NOT NULL,
                duration_seconds INTEGER,
                is_short TINYINT(1),
                video_type VARCHAR(50),
                tags TEXT,
                category_id VARCHAR(50),
                last_updated VARCHAR(50),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                view_count INTEGER,
                like_count INTEGER,
                comment_count INTEGER,
                engagement_rate FLOAT,
                recorded_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                recommendation_date VARCHAR(50) NOT NULL,
                recommended_type VARCHAR(100),
                recommended_topic TEXT,
                reasoning TEXT,
                predicted_performance VARCHAR(255),
                title_suggestions TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Migración: agregar columna title_suggestions si no existe (para BDs existentes)
        try:
            cursor.execute("""
                ALTER TABLE recommendations ADD COLUMN title_suggestions TEXT AFTER predicted_performance
            """)
            self.conn.commit()
        except Exception:
            log.debug("Migración: columna title_suggestions ya existe")

        # Migración: agregar columna is_competitor a channels (para BDs existentes)
        try:
            cursor.execute("""
                ALTER TABLE channels ADD COLUMN is_competitor TINYINT(1) NOT NULL DEFAULT 0 AFTER view_count
            """)
            self.conn.commit()
        except Exception:
            log.debug("Migración: columna is_competitor ya existe")

        # Migración: agregar content_category a videos (para BDs existentes — 12.2)
        try:
            cursor.execute("""
                ALTER TABLE videos ADD COLUMN content_category VARCHAR(50) DEFAULT NULL AFTER category_id
            """)
            self.conn.commit()
        except Exception:
            log.debug("Migración: columna content_category ya existe")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS virality_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                virality_score FLOAT NOT NULL,
                predicted_at VARCHAR(50) NOT NULL,
                model_features TEXT,
                UNIQUE KEY uq_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS view_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                predicted_views BIGINT NOT NULL,
                predicted_low   BIGINT NOT NULL,
                predicted_high  BIGINT NOT NULL,
                predicted_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_vp_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_analytics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                avg_view_duration_seconds FLOAT DEFAULT 0,
                avg_view_percentage FLOAT DEFAULT 0,
                estimated_minutes_watched BIGINT DEFAULT 0,
                shares INT DEFAULT 0,
                subscribers_gained INT DEFAULT 0,
                impressions BIGINT DEFAULT 0,
                impression_ctr FLOAT DEFAULT 0,
                recorded_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_va_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_sources (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                source_type VARCHAR(100) NOT NULL,
                source_label VARCHAR(200),
                views BIGINT DEFAULT 0,
                estimated_minutes BIGINT DEFAULT 0,
                recorded_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_ts_channel_source (channel_id, source_type),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                recommendation_date VARCHAR(50) NOT NULL,
                recommended_type VARCHAR(50),
                video_id VARCHAR(255) DEFAULT NULL,
                video_type VARCHAR(50) DEFAULT NULL,
                followed_recommendation TINYINT(1) DEFAULT NULL,
                view_count BIGINT DEFAULT NULL,
                channel_avg_at_time FLOAT DEFAULT NULL,
                performance_ratio FLOAT DEFAULT NULL,
                performance_label VARCHAR(50) DEFAULT NULL,
                linked_at DATETIME DEFAULT NULL,
                created_at DATETIME NOT NULL,
                UNIQUE KEY uq_rr_channel_date (channel_id, recommendation_date(20)),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weekly_plans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                week_start_date VARCHAR(20) NOT NULL,
                plan_json TEXT NOT NULL,
                strategy TEXT,
                generated_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_wp_channel_week (channel_id, week_start_date),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS script_outlines (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                recommendation_id INT,
                video_type VARCHAR(100),
                topic TEXT,
                title TEXT,
                outline_text TEXT NOT NULL,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seo_content (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                recommendation_id INT,
                title TEXT NOT NULL,
                seo_description TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                hashtags_json TEXT,
                related_videos_json TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retention_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                predicted_retention FLOAT NOT NULL,
                predicted_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_rp_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channel_health_reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                health_score INT NOT NULL,
                metrics_json TEXT NOT NULL,
                ai_diagnosis TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS competitor_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                channel_name VARCHAR(500),
                video_title VARCHAR(1000),
                view_count BIGINT NOT NULL,
                competitor_avg_views FLOAT NOT NULL,
                ratio FLOAT NOT NULL,
                ai_analysis TEXT,
                notified TINYINT(1) NOT NULL DEFAULT 0,
                created_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_ca_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Índices (ignorar error si ya existen)
        for sql in [
            "CREATE INDEX idx_videos_channel ON videos(channel_id)",
            "CREATE INDEX idx_metrics_video ON video_metrics(video_id, recorded_at(20))",
            "CREATE INDEX idx_recommendations_channel ON recommendations(channel_id, recommendation_date(20))",
        ]:
            try:
                cursor.execute(sql)
            except pymysql.err.OperationalError:
                log.debug("Índice ya existe: %s", sql[:60])

        self.conn.commit()

    def save_channel_data(self, channel_data: dict):
        """Guarda o actualiza información de un canal (propio o competidor)"""
        cursor = self.conn.cursor()
        channel_data['last_updated'] = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO channels
            (channel_id, channel_name, description, subscriber_count,
             video_count, view_count, is_competitor, created_at, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                channel_name = VALUES(channel_name),
                description = VALUES(description),
                subscriber_count = VALUES(subscriber_count),
                video_count = VALUES(video_count),
                view_count = VALUES(view_count),
                is_competitor = VALUES(is_competitor),
                last_updated = VALUES(last_updated)
        """, (
            channel_data['channel_id'],
            channel_data['channel_name'],
            channel_data['description'],
            channel_data['subscriber_count'],
            channel_data['video_count'],
            channel_data['view_count'],
            channel_data.get('is_competitor', 0),
            channel_data['created_at'],
            channel_data['last_updated']
        ))

        self.conn.commit()

    def save_videos_data(self, videos_df: pd.DataFrame):
        """Guarda o actualiza datos de videos y sus métricas"""
        cursor = self.conn.cursor()
        current_time = datetime.now().isoformat()

        for _, video in videos_df.iterrows():
            cursor.execute("""
                INSERT INTO videos
                (video_id, channel_id, title, description, published_at,
                 duration_seconds, is_short, video_type, tags, category_id, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    description = VALUES(description),
                    duration_seconds = VALUES(duration_seconds),
                    is_short = VALUES(is_short),
                    video_type = VALUES(video_type),
                    tags = VALUES(tags),
                    category_id = VALUES(category_id),
                    last_updated = VALUES(last_updated)
            """, (
                video['video_id'],
                video['channel_id'],
                video['title'],
                video['description'],
                video['published_at'].isoformat() if pd.notna(video['published_at']) else None,
                video['duration_seconds'],
                video['is_short'],
                video['video_type'],
                video['tags'],
                video['category_id'],
                current_time
            ))

            cursor.execute("""
                INSERT INTO video_metrics
                (video_id, view_count, like_count, comment_count,
                 engagement_rate, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                video['video_id'],
                video['view_count'],
                video['like_count'],
                video['comment_count'],
                video['engagement_rate'],
                current_time
            ))

        self.conn.commit()

    def get_all_videos(self, channel_id: Optional[str] = None) -> pd.DataFrame:
        """Obtiene todos los videos con sus últimas métricas"""
        query = """
            SELECT
                v.*,
                c.channel_name AS channel_title,
                c.subscriber_count,
                m.view_count,
                m.like_count,
                m.comment_count,
                m.engagement_rate,
                m.recorded_at
            FROM videos v
            LEFT JOIN channels c ON v.channel_id = c.channel_id
            LEFT JOIN (
                SELECT video_id, view_count, like_count, comment_count,
                       engagement_rate, recorded_at,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) as rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
        """

        cursor = self.conn.cursor()
        if channel_id:
            query += " WHERE v.channel_id = %s"
            cursor.execute(query, (channel_id,))
        else:
            cursor.execute(query)

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)

        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            if 'recorded_at' in df.columns:
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])

        return df

    def get_video_metrics_history(self, video_id: str) -> pd.DataFrame:
        """Obtiene el histórico de métricas de un video"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM video_metrics
            WHERE video_id = %s
            ORDER BY recorded_at
        """, (video_id,))

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df['recorded_at'] = pd.to_datetime(df['recorded_at'])

        return df

    def get_channel_metrics_history(self, channel_id: str) -> pd.DataFrame:
        """
        Retorna la evolución histórica diaria del canal.
        Deduplica múltiples ejecuciones del mismo día tomando la última snapshot.
        Columnas: snapshot_date, total_views, total_likes, avg_engagement, videos_tracked
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                DATE(latest.recorded_at) AS snapshot_date,
                SUM(latest.view_count)    AS total_views,
                SUM(latest.like_count)    AS total_likes,
                AVG(latest.engagement_rate) AS avg_engagement,
                COUNT(DISTINCT latest.video_id) AS videos_tracked
            FROM (
                SELECT
                    vm.video_id,
                    vm.view_count,
                    vm.like_count,
                    vm.engagement_rate,
                    vm.recorded_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY vm.video_id, DATE(vm.recorded_at)
                        ORDER BY vm.recorded_at DESC
                    ) AS rn
                FROM video_metrics vm
                JOIN videos v ON vm.video_id = v.video_id
                WHERE v.channel_id = %s
            ) latest
            WHERE latest.rn = 1
            GROUP BY DATE(latest.recorded_at)
            ORDER BY snapshot_date
        """, (channel_id,))

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

        return df

    def get_top_videos_growth(self, channel_id: str, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna la evolución de vistas de los N videos más vistos del canal.
        Deduplica por día tomando el MAX(view_count) del día.
        Columnas: video_id, title, video_type, snapshot_date, view_count
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                vm.video_id,
                v.title,
                v.video_type,
                DATE(vm.recorded_at) AS snapshot_date,
                MAX(vm.view_count)   AS view_count
            FROM video_metrics vm
            JOIN videos v ON vm.video_id = v.video_id
            JOIN (
                SELECT vm2.video_id
                FROM video_metrics vm2
                JOIN videos v2 ON vm2.video_id = v2.video_id
                WHERE v2.channel_id = %s
                GROUP BY vm2.video_id
                ORDER BY MAX(vm2.view_count) DESC
                LIMIT %s
            ) top_videos ON vm.video_id = top_videos.video_id
            WHERE v.channel_id = %s
            GROUP BY vm.video_id, v.title, v.video_type, DATE(vm.recorded_at)
            ORDER BY vm.video_id, snapshot_date
        """, (channel_id, top_n, channel_id))

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

        return df

    def save_recommendation(self, channel_id: str, recommendation: dict):
        """Guarda una recomendación generada"""
        cursor = self.conn.cursor()

        # Serializar sugerencias de título como JSON
        title_suggestions = recommendation.get('title_suggestions')
        title_suggestions_json = json.dumps(title_suggestions, ensure_ascii=False) if title_suggestions else None

        cursor.execute("""
            INSERT INTO recommendations
            (channel_id, recommendation_date, recommended_type,
             recommended_topic, reasoning, predicted_performance,
             title_suggestions, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            channel_id,
            recommendation.get('recommendation_date', datetime.now().date().isoformat()),
            recommendation.get('recommended_type'),
            recommendation.get('recommended_topic'),
            recommendation.get('reasoning'),
            recommendation.get('predicted_performance'),
            title_suggestions_json,
            datetime.now().isoformat()
        ))

        self.conn.commit()

    def get_recommendations(self, channel_id: str, days: int = 30) -> pd.DataFrame:
        """Obtiene recomendaciones recientes"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM recommendations
            WHERE channel_id = %s
            AND DATE(created_at) >= DATE(DATE_SUB(NOW(), INTERVAL %s DAY))
            ORDER BY created_at DESC
        """, (channel_id, days))

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df

    def get_performance_summary(self, channel_id: str) -> dict:
        """Obtiene un resumen del performance del canal"""
        df = self.get_all_videos(channel_id)

        if df.empty:
            return {}

        return {
            'total_videos': len(df),
            'total_shorts': len(df[df['is_short'] == True]),
            'total_long_videos': len(df[df['is_short'] == False]),
            'total_views': df['view_count'].sum(),
            'avg_views_per_video': df['view_count'].mean(),
            'avg_engagement_rate': df['engagement_rate'].mean(),
            'best_performing_video': df.loc[df['view_count'].idxmax(), 'title'],
            'best_performing_views': df['view_count'].max(),
            'shorts_avg_views': df[df['is_short'] == True]['view_count'].mean() if len(df[df['is_short'] == True]) > 0 else 0,
            'long_videos_avg_views': df[df['is_short'] == False]['view_count'].mean() if len(df[df['is_short'] == False]) > 0 else 0,
        }

    def save_virality_predictions(self, predictions_df: pd.DataFrame):
        """Guarda o actualiza los scores de viralidad para un lote de videos."""
        cursor = self.conn.cursor()
        predicted_at = datetime.now().isoformat()

        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO virality_predictions
                (video_id, channel_id, virality_score, predicted_at, model_features)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    virality_score = VALUES(virality_score),
                    predicted_at   = VALUES(predicted_at),
                    model_features = VALUES(model_features)
            """, (
                row['video_id'],
                row['channel_id'],
                float(row['virality_score']),
                predicted_at,
                row.get('model_features', None),
            ))

        self.conn.commit()

    def get_virality_predictions(self, channel_id: str) -> pd.DataFrame:
        """Obtiene los scores de viralidad de todos los videos de un canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                vp.video_id,
                vp.virality_score,
                vp.predicted_at,
                v.title,
                v.video_type,
                v.published_at,
                v.duration_seconds
            FROM virality_predictions vp
            JOIN videos v ON vp.video_id = v.video_id
            WHERE vp.channel_id = %s
            ORDER BY vp.virality_score DESC
        """, (channel_id,))

        rows = cursor.fetchall()
        return pd.DataFrame(rows)

    def save_view_predictions(self, predictions_df: pd.DataFrame):
        """Guarda o actualiza las predicciones de vistas para un lote de videos."""
        cursor = self.conn.cursor()
        predicted_at = datetime.now().isoformat()

        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO view_predictions
                (video_id, channel_id, predicted_views, predicted_low, predicted_high, predicted_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    predicted_views = VALUES(predicted_views),
                    predicted_low   = VALUES(predicted_low),
                    predicted_high  = VALUES(predicted_high),
                    predicted_at    = VALUES(predicted_at)
            """, (
                row['video_id'],
                row['channel_id'],
                int(row['predicted_views']),
                int(row['predicted_low']),
                int(row['predicted_high']),
                predicted_at,
            ))

        self.conn.commit()

    def get_view_predictions(self, channel_id: str) -> pd.DataFrame:
        """Obtiene las predicciones de vistas de todos los videos de un canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                vp.video_id,
                vp.predicted_views,
                vp.predicted_low,
                vp.predicted_high,
                vp.predicted_at,
                v.title,
                v.video_type,
                v.published_at
            FROM view_predictions vp
            JOIN videos v ON vp.video_id = v.video_id
            WHERE vp.channel_id = %s
            ORDER BY vp.predicted_views DESC
        """, (channel_id,))

        rows = cursor.fetchall()
        return pd.DataFrame(rows)

    def save_video_analytics(self, analytics_df: pd.DataFrame):
        """Guarda o actualiza las métricas avanzadas (Analytics API) por video."""
        cursor = self.conn.cursor()
        recorded_at = datetime.now().isoformat()

        for _, row in analytics_df.iterrows():
            cursor.execute("""
                INSERT INTO video_analytics
                (video_id, channel_id, avg_view_duration_seconds, avg_view_percentage,
                 estimated_minutes_watched, shares, subscribers_gained,
                 impressions, impression_ctr, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    avg_view_duration_seconds = VALUES(avg_view_duration_seconds),
                    avg_view_percentage       = VALUES(avg_view_percentage),
                    estimated_minutes_watched = VALUES(estimated_minutes_watched),
                    shares                    = VALUES(shares),
                    subscribers_gained        = VALUES(subscribers_gained),
                    impressions               = VALUES(impressions),
                    impression_ctr            = VALUES(impression_ctr),
                    recorded_at               = VALUES(recorded_at)
            """, (
                row['video_id'],
                row['channel_id'],
                float(row.get('avg_view_duration_seconds', 0)),
                float(row.get('avg_view_percentage', 0)),
                int(row.get('estimated_minutes_watched', 0)),
                int(row.get('shares', 0)),
                int(row.get('subscribers_gained', 0)),
                int(row.get('impressions', 0)),
                float(row.get('impression_ctr', 0)),
                recorded_at,
            ))

        self.conn.commit()

    def get_video_analytics(self, channel_id: str) -> pd.DataFrame:
        """Obtiene métricas avanzadas de los videos de un canal, unidas con title/video_type."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                va.video_id,
                v.title,
                v.video_type,
                v.published_at,
                va.avg_view_duration_seconds,
                va.avg_view_percentage,
                va.estimated_minutes_watched,
                va.shares,
                va.subscribers_gained,
                va.impressions,
                va.impression_ctr,
                va.recorded_at
            FROM video_analytics va
            JOIN videos v ON va.video_id = v.video_id
            WHERE va.channel_id = %s
            ORDER BY va.avg_view_percentage DESC
        """, (channel_id,))

        rows = cursor.fetchall()
        return pd.DataFrame(rows)

    def save_traffic_sources(self, traffic_df: pd.DataFrame):
        """Guarda o actualiza las fuentes de tráfico del canal."""
        cursor = self.conn.cursor()
        recorded_at = datetime.now().isoformat()

        for _, row in traffic_df.iterrows():
            cursor.execute("""
                INSERT INTO traffic_sources
                (channel_id, source_type, source_label, views, estimated_minutes, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    source_label      = VALUES(source_label),
                    views             = VALUES(views),
                    estimated_minutes = VALUES(estimated_minutes),
                    recorded_at       = VALUES(recorded_at)
            """, (
                row['channel_id'],
                row['source_type'],
                row.get('source_label', row['source_type']),
                int(row['views']),
                int(row['estimated_minutes']),
                recorded_at,
            ))

        self.conn.commit()

    def get_traffic_sources(self, channel_id: str) -> pd.DataFrame:
        """Obtiene las fuentes de tráfico del canal ordenadas por vistas."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_type, source_label, views, estimated_minutes, recorded_at
            FROM traffic_sources
            WHERE channel_id = %s
            ORDER BY views DESC
        """, (channel_id,))

        rows = cursor.fetchall()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Retroalimentación del ciclo de aprendizaje (Mejora 4.1)
    # ------------------------------------------------------------------

    def save_recommendation_result(
        self,
        channel_id: str,
        recommendation_date: str,
        recommended_type: str,
    ) -> None:
        """
        Crea un registro de resultado de recomendación cuando se genera una nueva.
        Ignora si ya existe un registro para esa fecha/canal.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT IGNORE INTO recommendation_results
            (channel_id, recommendation_date, recommended_type, created_at)
            VALUES (%s, %s, %s, %s)
        """, (channel_id, recommendation_date, recommended_type, datetime.now().isoformat()))
        self.conn.commit()

    def link_video_to_recommendation(
        self,
        channel_id: str,
        channel_avg_views: float,
    ) -> int:
        """
        Vincula el primer video publicado tras cada recomendación sin vincular.
        Calcula performance_ratio vs el promedio del canal en ese momento.
        Solo procesa recomendaciones con al menos 1 día de antigüedad.

        Returns:
            Número de recomendaciones vinculadas en esta llamada.
        """
        from datetime import timedelta
        cursor = self.conn.cursor()

        # Recomendaciones sin video vinculado y con al menos 1 día de antigüedad
        cursor.execute("""
            SELECT id, recommendation_date, recommended_type
            FROM recommendation_results
            WHERE channel_id = %s
              AND video_id IS NULL
              AND recommendation_date < %s
            ORDER BY recommendation_date ASC
        """, (channel_id, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')))

        unlinked = cursor.fetchall()
        linked = 0

        for rec in unlinked:
            # Primer video publicado DESPUÉS de la fecha de la recomendación
            # view_count vive en video_metrics (la snapshot más reciente)
            cursor.execute("""
                SELECT v.video_id, v.video_type,
                       COALESCE(vm.view_count, 0) AS view_count
                FROM videos v
                LEFT JOIN video_metrics vm ON vm.video_id = v.video_id
                    AND vm.recorded_at = (
                        SELECT MAX(vm2.recorded_at)
                        FROM video_metrics vm2
                        WHERE vm2.video_id = v.video_id
                    )
                WHERE v.channel_id = %s AND v.published_at > %s
                ORDER BY v.published_at ASC
                LIMIT 1
            """, (channel_id, rec['recommendation_date']))

            video = cursor.fetchone()
            if not video:
                continue

            ratio = (video['view_count'] / channel_avg_views
                     if channel_avg_views and channel_avg_views > 0 else None)
            label = None
            if ratio is not None:
                label = ('above_average' if ratio >= 1.2
                         else 'below_average' if ratio < 0.8
                         else 'average')

            followed = 1 if video['video_type'] == rec['recommended_type'] else 0

            cursor.execute("""
                UPDATE recommendation_results
                SET video_id              = %s,
                    video_type            = %s,
                    followed_recommendation = %s,
                    view_count            = %s,
                    channel_avg_at_time   = %s,
                    performance_ratio     = %s,
                    performance_label     = %s,
                    linked_at             = %s
                WHERE id = %s
            """, (
                video['video_id'], video['video_type'], followed,
                video['view_count'], channel_avg_views,
                ratio, label, datetime.now().isoformat(),
                rec['id'],
            ))
            linked += 1

        self.conn.commit()
        return linked

    def get_recommendation_results(
        self,
        channel_id: str,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Retorna los resultados de recomendaciones más recientes para el canal,
        incluyendo el título del video vinculado si existe.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rr.*, v.title AS video_title
            FROM recommendation_results rr
            LEFT JOIN videos v ON rr.video_id = v.video_id
            WHERE rr.channel_id = %s
            ORDER BY rr.recommendation_date DESC
            LIMIT %s
        """, (channel_id, limit))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_weekly_plan(
        self,
        channel_id: str,
        week_start_date: str,
        plan_json: str,
        strategy: str,
        generated_at: str,
    ) -> None:
        """
        Guarda (o actualiza) un plan semanal en la BD.

        Args:
            channel_id: ID del canal.
            week_start_date: Fecha de inicio de la semana (YYYY-MM-DD).
            plan_json: JSON serializado con la lista de días del plan.
            strategy: Texto de estrategia general generado por Claude.
            generated_at: ISO timestamp de generación.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO weekly_plans
                (channel_id, week_start_date, plan_json, strategy, generated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                plan_json    = VALUES(plan_json),
                strategy     = VALUES(strategy),
                generated_at = VALUES(generated_at)
            """,
            (channel_id, week_start_date, plan_json, strategy, generated_at),
        )
        self.conn.commit()

    def get_weekly_plans(self, channel_id: str, limit: int = 4) -> pd.DataFrame:
        """
        Retorna los planes semanales más recientes de un canal.

        Returns:
            DataFrame con columnas: week_start_date, plan_json, strategy, generated_at
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT week_start_date, plan_json, strategy, generated_at
            FROM weekly_plans
            WHERE channel_id = %s
            ORDER BY week_start_date DESC
            LIMIT %s
            """,
            (channel_id, limit),
        )
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Script Outlines
    # ------------------------------------------------------------------

    def save_script_outline(self, channel_id: str, recommendation_id: int, outline_data: dict):
        """Guarda un guion/outline generado para una recomendacion."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO script_outlines
            (channel_id, recommendation_id, video_type, topic, title, outline_text, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                channel_id,
                recommendation_id,
                outline_data.get('video_type', ''),
                outline_data.get('topic', ''),
                outline_data.get('title', ''),
                outline_data.get('outline_text', ''),
                outline_data.get('created_at', datetime.now().isoformat()),
            ),
        )
        self.conn.commit()

    def get_script_outline(self, recommendation_id: int) -> dict | None:
        """
        Retorna el outline asociado a una recomendacion, o None si no existe.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, video_type, topic, title, outline_text, created_at
            FROM script_outlines
            WHERE recommendation_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (recommendation_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'id': row['id'],
            'video_type': row['video_type'],
            'topic': row['topic'],
            'title': row['title'],
            'outline_text': row['outline_text'],
            'created_at': row['created_at'],
        }

    # ------------------------------------------------------------------
    # SEO Content (Mejora 9.2 + 9.3)
    # ------------------------------------------------------------------

    def save_seo_content(self, channel_id: str, recommendation_id: int | None,
                         seo_data: dict):
        """Guarda contenido SEO generado (descripción + tags) para una recomendación."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO seo_content
            (channel_id, recommendation_id, title, seo_description,
             tags_json, hashtags_json, related_videos_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                channel_id,
                recommendation_id,
                seo_data.get('title', ''),
                seo_data.get('seo_description', ''),
                json.dumps(seo_data.get('tags', []), ensure_ascii=False),
                json.dumps(seo_data.get('hashtags', []), ensure_ascii=False),
                json.dumps(seo_data.get('related_videos', []), ensure_ascii=False),
                seo_data.get('created_at', datetime.now().isoformat()),
            ),
        )
        self.conn.commit()

    def get_seo_content(self, recommendation_id: int) -> dict | None:
        """
        Retorna el contenido SEO asociado a una recomendación, o None si no existe.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, title, seo_description, tags_json, hashtags_json,
                   related_videos_json, created_at
            FROM seo_content
            WHERE recommendation_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (recommendation_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'id': row['id'],
            'title': row['title'],
            'seo_description': row['seo_description'],
            'tags': json.loads(row['tags_json']) if row['tags_json'] else [],
            'hashtags': json.loads(row['hashtags_json']) if row['hashtags_json'] else [],
            'related_videos': json.loads(row['related_videos_json']) if row['related_videos_json'] else [],
            'created_at': row['created_at'],
        }

    def get_related_videos_by_keywords(
        self, channel_id: str, keywords: list[str], limit: int = 5
    ) -> list[dict]:
        """
        Busca videos del canal cuyo título contenga alguna de las keywords.
        Retorna los más vistos para sugerir como enlaces relacionados en la descripción SEO.
        """
        if not keywords:
            return []
        cursor = self.conn.cursor()
        # Construir condición LIKE para cada keyword
        like_clauses = ' OR '.join(['v.title LIKE %s'] * len(keywords))
        params: list = [channel_id] + [f'%{kw}%' for kw in keywords]
        params.append(limit)
        cursor.execute(f"""
            SELECT v.video_id, v.title, m.view_count
            FROM videos v
            LEFT JOIN (
                SELECT video_id, view_count,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
            WHERE v.channel_id = %s AND ({like_clauses})
            ORDER BY COALESCE(m.view_count, 0) DESC
            LIMIT %s
        """, params)
        rows = cursor.fetchall()
        return [
            {
                'video_id': r['video_id'],
                'title': r['title'],
                'url': f"https://youtu.be/{r['video_id']}",
                'view_count': r.get('view_count') or 0,
            }
            for r in rows
        ]

    def get_top_tags_from_channel(self, channel_id: str, top_n: int = 30) -> list[str]:
        """
        Extrae los tags más frecuentes de los videos más exitosos del canal.
        Retorna lista de tags ordenados por frecuencia.
        """
        from collections import Counter

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.tags
            FROM videos v
            LEFT JOIN (
                SELECT video_id, view_count,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
            WHERE v.channel_id = %s AND v.tags IS NOT NULL AND v.tags != ''
            ORDER BY COALESCE(m.view_count, 0) DESC
            LIMIT 20
        """, (channel_id,))
        rows = cursor.fetchall()

        tag_counter: Counter = Counter()
        for row in rows:
            if row['tags']:
                for tag in row['tags'].split(','):
                    tag = tag.strip()
                    if tag:
                        tag_counter[tag.lower()] += 1

        return [tag for tag, _ in tag_counter.most_common(top_n)]

    # ------------------------------------------------------------------
    # Retention Predictions (Mejora 12.1)
    # ------------------------------------------------------------------

    def save_retention_predictions(self, predictions_df: pd.DataFrame):
        """Guarda predicciones de retención para los videos."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO retention_predictions
                (video_id, channel_id, predicted_retention, predicted_at)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    predicted_retention = VALUES(predicted_retention),
                    predicted_at = VALUES(predicted_at)
            """, (
                row['video_id'],
                row['channel_id'],
                float(row['predicted_retention']),
                now,
            ))
        self.conn.commit()

    def get_retention_predictions(self, channel_id: str) -> pd.DataFrame:
        """Obtiene predicciones de retención con datos del video."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT rp.video_id, rp.predicted_retention, rp.predicted_at,
                   v.title, v.video_type, v.published_at, v.duration_seconds
            FROM retention_predictions rp
            JOIN videos v ON rp.video_id = v.video_id
            WHERE rp.channel_id = %s
            ORDER BY rp.predicted_retention DESC
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Content Classification (Mejora 12.2)
    # ------------------------------------------------------------------

    def save_content_categories(self, categories: dict[str, str]):
        """Actualiza la categoría de contenido para un lote de videos."""
        cursor = self.conn.cursor()
        for video_id, category in categories.items():
            cursor.execute(
                "UPDATE videos SET content_category = %s WHERE video_id = %s",
                (category, video_id),
            )
        self.conn.commit()

    def get_videos_without_category(self, channel_id: str) -> pd.DataFrame:
        """Retorna videos del canal sin categoría asignada."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT video_id, title, tags, description
            FROM videos
            WHERE channel_id = %s AND (content_category IS NULL OR content_category = '')
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_performance_by_category(self, channel_id: str) -> pd.DataFrame:
        """
        Retorna métricas de performance agrupadas por categoría de contenido.
        Incluye avg views, avg engagement, conteo de videos.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.content_category,
                   COUNT(*) AS video_count,
                   ROUND(AVG(m.view_count), 0) AS avg_views,
                   ROUND(AVG(m.engagement_rate), 2) AS avg_engagement,
                   ROUND(AVG(va.avg_view_percentage), 1) AS avg_retention
            FROM videos v
            LEFT JOIN (
                SELECT video_id, view_count, engagement_rate,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
            LEFT JOIN video_analytics va ON v.video_id = va.video_id
            WHERE v.channel_id = %s
              AND v.content_category IS NOT NULL
              AND v.content_category != ''
            GROUP BY v.content_category
            ORDER BY avg_views DESC
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Late Bloomer Detection (Mejora 12.3)
    # ------------------------------------------------------------------

    def get_videos_with_snapshot_counts(self, channel_id: str,
                                        min_snapshots: int = 3) -> pd.DataFrame:
        """Retorna videos del canal que tengan al menos N snapshots en video_metrics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.video_id, v.title, v.video_type, v.published_at,
                   COUNT(vm.id) AS snapshot_count
            FROM videos v
            JOIN video_metrics vm ON v.video_id = vm.video_id
            WHERE v.channel_id = %s
            GROUP BY v.video_id, v.title, v.video_type, v.published_at
            HAVING snapshot_count >= %s
            ORDER BY v.published_at DESC
        """, (channel_id, min_snapshots))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Channel Health Reports
    # ------------------------------------------------------------------

    def save_health_report(self, channel_id: str, health_score: int,
                           metrics_json: str, ai_diagnosis: str | None = None):
        """Guarda un reporte de salud del canal."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO channel_health_reports
            (channel_id, health_score, metrics_json, ai_diagnosis, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (channel_id, health_score, metrics_json, ai_diagnosis,
             datetime.now().isoformat()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_latest_health_report(self, channel_id: str) -> dict | None:
        """Retorna el reporte de salud más reciente (últimas 24 h) o None."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, health_score, metrics_json, ai_diagnosis, created_at
            FROM channel_health_reports
            WHERE channel_id = %s
              AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            ORDER BY id DESC
            LIMIT 1
            """,
            (channel_id,),
        )
        row = cursor.fetchone()
        return row if row else None

    def update_health_diagnosis(self, report_id: int, ai_diagnosis: str):
        """Actualiza el diagnóstico IA de un reporte existente."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE channel_health_reports SET ai_diagnosis = %s WHERE id = %s",
            (ai_diagnosis, report_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Competidores (Mejora 7.1)
    # ------------------------------------------------------------------

    def get_competitor_channels(self) -> pd.DataFrame:
        """Retorna todos los canales marcados como competidores."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT channel_id, channel_name, subscriber_count, video_count,
                   view_count, created_at, last_updated
            FROM channels
            WHERE is_competitor = 1
            ORDER BY subscriber_count DESC
        """)
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_own_channels(self) -> pd.DataFrame:
        """Retorna todos los canales propios (no competidores)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT channel_id, channel_name, subscriber_count, video_count,
                   view_count, created_at, last_updated
            FROM channels
            WHERE is_competitor = 0
            ORDER BY subscriber_count DESC
        """)
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_all_videos_with_competitor_flag(
        self, channel_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Obtiene videos con info del canal incluyendo is_competitor.
        Si se pasa channel_ids, filtra solo esos canales.
        """
        cursor = self.conn.cursor()
        query = """
            SELECT
                v.*,
                c.channel_name AS channel_title,
                c.subscriber_count,
                c.is_competitor,
                m.view_count,
                m.like_count,
                m.comment_count,
                m.engagement_rate,
                m.recorded_at
            FROM videos v
            LEFT JOIN channels c ON v.channel_id = c.channel_id
            LEFT JOIN (
                SELECT video_id, view_count, like_count, comment_count,
                       engagement_rate, recorded_at,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) as rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
        """
        if channel_ids:
            placeholders = ','.join(['%s'] * len(channel_ids))
            query += f" WHERE v.channel_id IN ({placeholders})"
            cursor.execute(query, channel_ids)
        else:
            cursor.execute(query)

        rows = cursor.fetchall()
        df = pd.DataFrame(rows)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            if 'recorded_at' in df.columns:
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])
        return df

    # ------------------------------------------------------------------
    # Alertas de Competidores (Mejora 7.2)
    # ------------------------------------------------------------------

    def get_recent_competitor_videos(self, days: int = 7) -> pd.DataFrame:
        """Retorna videos de competidores publicados en los últimos N días,
        junto con su vista actual y el promedio de vistas del canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                v.video_id,
                v.channel_id,
                v.title,
                v.published_at,
                v.video_type,
                c.channel_name,
                m.view_count,
                m.like_count,
                m.engagement_rate,
                m.recorded_at AS metric_recorded_at,
                ch_avg.avg_views AS competitor_avg_views
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id AND c.is_competitor = 1
            LEFT JOIN (
                SELECT video_id, view_count, like_count, engagement_rate, recorded_at,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
            LEFT JOIN (
                SELECT v2.channel_id, AVG(m2.view_count) AS avg_views
                FROM videos v2
                JOIN (
                    SELECT video_id, view_count,
                           ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                    FROM video_metrics
                ) m2 ON v2.video_id = m2.video_id AND m2.rn = 1
                JOIN channels c2 ON v2.channel_id = c2.channel_id AND c2.is_competitor = 1
                GROUP BY v2.channel_id
            ) ch_avg ON v.channel_id = ch_avg.channel_id
            WHERE v.published_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
              AND c.is_competitor = 1
            ORDER BY m.view_count DESC
        """, (days,))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            if 'metric_recorded_at' in df.columns:
                df['metric_recorded_at'] = pd.to_datetime(df['metric_recorded_at'])
        return df

    def save_competitor_alert(self, alert: dict):
        """Guarda una alerta de competidor. Ignora duplicados (UNIQUE en video_id)."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO competitor_alerts
                (video_id, channel_id, channel_name, video_title, view_count,
                 competitor_avg_views, ratio, ai_analysis, notified, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                alert['video_id'], alert['channel_id'], alert['channel_name'],
                alert['video_title'], alert['view_count'],
                alert['competitor_avg_views'], alert['ratio'],
                alert.get('ai_analysis', ''), alert.get('notified', 0),
                datetime.now().isoformat(),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            if 'Duplicate' in str(e):
                log.debug("Alerta ya existe para video %s", alert['video_id'])
                return False
            raise

    def get_competitor_alerts(self, limit: int = 20) -> pd.DataFrame:
        """Retorna las últimas alertas de competidores."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, video_id, channel_id, channel_name, video_title,
                   view_count, competitor_avg_views, ratio, ai_analysis,
                   notified, created_at
            FROM competitor_alerts
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def is_alert_already_sent(self, video_id: str) -> bool:
        """Verifica si ya se envió una alerta para este video."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM competitor_alerts WHERE video_id = %s LIMIT 1",
            (video_id,),
        )
        return cursor.fetchone() is not None

    def close(self):
        """Cierra la conexión a la base de datos"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
