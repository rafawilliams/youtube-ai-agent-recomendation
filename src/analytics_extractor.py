"""
Extractor de métricas avanzadas via YouTube Analytics API v2.

Requiere OAuth 2.0 — solo funciona para canales propios (el usuario que
autoriza debe ser propietario o gestor del canal).

Configuración mínima:
  1. En Google Cloud Console habilita "YouTube Analytics API".
  2. Crea credenciales OAuth 2.0 (Aplicación de escritorio).
  3. Descarga el JSON y guárdalo como credentials.json en la raíz del proyecto.
  4. La primera ejecución abrirá el navegador para autorizar el acceso.
     El token se persiste en token.json para ejecuciones futuras.
"""
import os
import logging
from datetime import datetime
import pandas as pd

from retry_config import retry_google_api

log = logging.getLogger(__name__)


@retry_google_api
def _execute_with_retry(request):
    """Ejecuta una solicitud de Analytics API con retry en errores transitorios."""
    return request.execute()


SCOPES      = ['https://www.googleapis.com/auth/yt-analytics.readonly']
CHANNEL_MINE = 'channel==MINE'

TRAFFIC_SOURCE_LABELS = {
    'YT_SEARCH':         'Búsqueda YouTube',
    'SUGGESTED':         'Videos Sugeridos',
    'BROWSE':            'Inicio / Explorar',
    'EXT_URL':           'Fuentes Externas',
    'NOTIFICATION':      'Notificaciones',
    'SUBSCRIBER':        'Feed de Suscriptores',
    'PLAYLIST':          'Listas de Reproducción',
    'NO_LINK_EMBEDDED':  'Video Incrustado',
    'NO_LINK_OTHER':     'Directo / Otros',
    'OTHER':             'Otros',
}


class YouTubeAnalyticsExtractor:
    """
    Extrae métricas avanzadas de YouTube Analytics API v2.

    Métricas de video:
      - avg_view_duration_seconds  → duración media vista (segundos)
      - avg_view_percentage        → retención media (% del video visto)
      - estimated_minutes_watched  → minutos totales vistos
      - shares                     → veces compartido
      - subscribers_gained         → suscriptores ganados desde ese video
      - impressions                → impresiones de miniatura
      - impression_ctr             → CTR de miniatura (0-100 %)

    Canal:
      - Fuentes de tráfico (traffic_sources)
    """

    def __init__(self, credentials_file: str = None, token_file: str = None):
        """
        Args:
            credentials_file: Ruta a credentials.json (por defecto 'credentials.json').
            token_file:        Ruta al token persistido (por defecto 'token.json').
        """
        self.credentials_file = credentials_file or os.getenv(
            'GOOGLE_CREDENTIALS_FILE', 'credentials.json'
        )
        self.token_file = token_file or os.getenv(
            'GOOGLE_TOKEN_FILE', 'token.json'
        )
        self._analytics = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """True si existe credentials.json o un token previo."""
        return os.path.exists(self.credentials_file) or os.path.exists(self.token_file)

    def authenticate(self) -> bool:
        """
        Ejecuta el flujo OAuth 2.0 si es necesario y construye el servicio.
        Retorna True si la autenticación fue exitosa.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            creds = None

            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_file):
                        log.warning(
                            "Analytics: '%s' no encontrado. Descárgalo desde Google Cloud "
                            "Console → Credenciales → OAuth 2.0 (Aplicación de escritorio).",
                            self.credentials_file,
                        )
                        return False
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file, SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                with open(self.token_file, 'w') as f:
                    f.write(creds.to_json())

            self._analytics = build('youtubeAnalytics', 'v2', credentials=creds)
            return True

        except Exception as e:
            log.warning("Analytics: Error de autenticación — %s", e)
            return False

    # ------------------------------------------------------------------
    # Video-level metrics
    # ------------------------------------------------------------------

    def get_video_analytics(
        self,
        channel_id: str,
        start_date: str = '2020-01-01',
    ) -> pd.DataFrame:
        """
        Retorna métricas de retención, engagement y alcance por video.

        Columnas:
            video_id, views, avg_view_duration_seconds, avg_view_percentage,
            estimated_minutes_watched, shares, subscribers_gained,
            impressions, impression_ctr, channel_id, recorded_at
        """
        if not self._analytics:
            return pd.DataFrame()

        end_date = datetime.now().strftime('%Y-%m-%d')

        # ── Query 1: retención y engagement ──────────────────────────────
        try:
            request = self._analytics.reports().query(
                ids=CHANNEL_MINE,
                startDate=start_date,
                endDate=end_date,
                metrics=(
                    'views,'
                    'estimatedMinutesWatched,'
                    'averageViewDuration,'
                    'averageViewPercentage,'
                    'shares,'
                    'subscribersGained'
                ),
                dimensions='video',
                sort='-views',
                maxResults=200,
            )
            resp = _execute_with_retry(request)

            rows = resp.get('rows', [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                'video_id', 'views', 'estimated_minutes_watched',
                'avg_view_duration_seconds', 'avg_view_percentage',
                'shares', 'subscribers_gained',
            ])
            df = df.astype({
                'views':                    int,
                'estimated_minutes_watched': int,
                'avg_view_duration_seconds': float,
                'avg_view_percentage':       float,
                'shares':                   int,
                'subscribers_gained':       int,
            })

        except Exception as e:
            log.error("Analytics: Error en métricas de video — %s", e)
            return pd.DataFrame()

        # ── Query 2: impresiones y CTR (grupo de métricas separado) ──────
        try:
            request2 = self._analytics.reports().query(
                ids=CHANNEL_MINE,
                startDate=start_date,
                endDate=end_date,
                metrics='impressions,impressionClickThroughRate',
                dimensions='video',
                sort='-impressions',
                maxResults=200,
            )
            resp2 = _execute_with_retry(request2)

            rows2 = resp2.get('rows', [])
            if rows2:
                df_ctr = pd.DataFrame(
                    rows2, columns=['video_id', 'impressions', 'impression_ctr']
                )
                df_ctr = df_ctr.astype({'impressions': int, 'impression_ctr': float})
                df = df.merge(df_ctr, on='video_id', how='left')

        except Exception as e:
            log.debug("CTR no disponible para este canal: %s", e)

        if 'impressions' not in df.columns:
            df['impressions'] = 0
        if 'impression_ctr' not in df.columns:
            df['impression_ctr'] = 0.0

        df['impressions'] = df['impressions'].fillna(0).astype(int)
        df['impression_ctr'] = df['impression_ctr'].fillna(0.0)

        df['channel_id'] = channel_id
        df['recorded_at'] = datetime.now().isoformat()
        return df

    # ------------------------------------------------------------------
    # Channel-level traffic sources
    # ------------------------------------------------------------------

    def get_traffic_sources(
        self,
        channel_id: str,
        start_date: str = '2020-01-01',
    ) -> pd.DataFrame:
        """
        Retorna las fuentes de tráfico del canal.

        Columnas:
            channel_id, source_type, source_label, views, estimated_minutes, recorded_at
        """
        if not self._analytics:
            return pd.DataFrame()

        end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            request = self._analytics.reports().query(
                ids=CHANNEL_MINE,
                startDate=start_date,
                endDate=end_date,
                metrics='views,estimatedMinutesWatched',
                dimensions='insightTrafficSourceType',
                sort='-views',
            )
            resp = _execute_with_retry(request)

            rows = resp.get('rows', [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['source_type', 'views', 'estimated_minutes'])
            df['source_label'] = df['source_type'].map(TRAFFIC_SOURCE_LABELS).fillna(df['source_type'])
            df['channel_id'] = channel_id
            df['recorded_at'] = datetime.now().isoformat()
            return df[['channel_id', 'source_type', 'source_label', 'views', 'estimated_minutes', 'recorded_at']]

        except Exception as e:
            log.error("Analytics: Error en fuentes de tráfico — %s", e)
            return pd.DataFrame()
