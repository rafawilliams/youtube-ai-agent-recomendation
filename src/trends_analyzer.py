"""
Analizador de tendencias de Google Trends para temas de YouTube.
Usa pytrends para obtener datos de popularidad de búsqueda en tiempo real.
"""
import time
import logging
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# ── Compatibilidad: pytrends 4.9.x usa 'method_whitelist' (eliminado en urllib3 ≥2.0)
# Este patch hace que Retry ignore el argumento obsoleto en lugar de lanzar TypeError.
try:
    from urllib3.util.retry import Retry as _Retry
    _orig_retry_init = _Retry.__init__

    def _compat_retry_init(self, *args, **kwargs):
        kwargs.pop('method_whitelist', None)
        _orig_retry_init(self, *args, **kwargs)

    _Retry.__init__ = _compat_retry_init
except Exception:
    pass  # si falla el patch, pytrends lanzará el error original en tiempo de uso


# Opciones de geografía
GEO_OPTIONS: dict[str, str] = {
    'Panamá':      'PA',
    'México':      'MX',
    'Colombia':    'CO',
    'Venezuela':   'VE',
    'Costa Rica':  'CR',
    'Argentina':   'AR',
    'España':      'ES',
    'Estados Unidos': 'US',
    'Mundial':     '',
}

# Opciones de período
TIMEFRAME_OPTIONS: dict[str, str] = {
    'Último mes':        'today 1-m',
    'Últimos 3 meses':   'today 3-m',
    'Último año':        'today 12-m',
    'Últimos 5 años':    'today 5-y',
}

MAX_KEYWORDS = 5   # límite de Google Trends


class TrendsAnalyzer:
    """
    Consulta Google Trends vía pytrends para comparar popularidad de temas.
    Provee interés en el tiempo, score promedio y consultas relacionadas.
    """

    def __init__(
        self,
        hl: str = 'es',
        tz: int = 300,           # UTC-5 Panamá
        timeout: tuple = (10, 25),
        retries: int = 2,
        backoff_factor: float = 0.5,
    ):
        from pytrends.request import TrendReq
        self._pt = TrendReq(
            hl=hl,
            tz=tz,
            timeout=timeout,
            retries=retries,
            backoff_factor=backoff_factor,
        )

    # ------------------------------------------------------------------
    # Consultas principales
    # ------------------------------------------------------------------

    def get_interest_over_time(
        self,
        keywords: list[str],
        geo: str = 'PA',
        timeframe: str = 'today 3-m',
    ) -> pd.DataFrame:
        """
        Retorna el interés en el tiempo (0–100) para cada keyword.

        Returns:
            DataFrame con columna por keyword e índice de fechas.
            Vacío si no hay datos o hay error.
        """
        kws = [k.strip() for k in keywords if k.strip()][:MAX_KEYWORDS]
        if not kws:
            return pd.DataFrame()

        try:
            self._pt.build_payload(kws, cat=0, timeframe=timeframe, geo=geo, gprop='')
            df = self._pt.interest_over_time()
            if df.empty:
                return pd.DataFrame()
            # Eliminar columna 'isPartial' si existe
            df = df.drop(columns=['isPartial'], errors='ignore')
            return df[kws]
        except Exception:
            return pd.DataFrame()

    def get_trend_scores(
        self,
        keywords: list[str],
        geo: str = 'PA',
        timeframe: str = 'today 3-m',
    ) -> dict[str, float]:
        """
        Retorna el score promedio de interés (0–100) por keyword en el período.

        Returns:
            dict {keyword: score_promedio}
        """
        df = self.get_interest_over_time(keywords, geo=geo, timeframe=timeframe)
        if df.empty:
            return {k: 0.0 for k in keywords}
        return {col: round(float(df[col].mean()), 1) for col in df.columns}

    def get_related_queries(
        self,
        keyword: str,
        geo: str = 'PA',
    ) -> dict[str, pd.DataFrame]:
        """
        Retorna las consultas relacionadas (top y en ascenso) para un keyword.

        Returns:
            {'top': DataFrame, 'rising': DataFrame}  — puede estar vacío.
        """
        try:
            self._pt.build_payload([keyword], cat=0, timeframe='today 3-m', geo=geo, gprop='')
            related = self._pt.related_queries()
            result = related.get(keyword, {})
            return {
                'top':    result.get('top',    pd.DataFrame()),
                'rising': result.get('rising', pd.DataFrame()),
            }
        except Exception:
            return {'top': pd.DataFrame(), 'rising': pd.DataFrame()}

    def get_interest_by_region(
        self,
        keywords: list[str],
        geo: str = 'PA',
        resolution: str = 'REGION',
    ) -> pd.DataFrame:
        """
        Retorna el interés desglosado por región/subregión.

        Returns:
            DataFrame indexado por geoName con columna por keyword.
        """
        kws = [k.strip() for k in keywords if k.strip()][:MAX_KEYWORDS]
        if not kws:
            return pd.DataFrame()

        try:
            self._pt.build_payload(kws, cat=0, timeframe='today 3-m', geo=geo, gprop='')
            df = self._pt.interest_by_region(resolution=resolution, inc_low_vol=False)
            return df
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Verifica que pytrends esté instalado."""
        try:
            import pytrends  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def peak_day(interest_df: pd.DataFrame) -> dict[str, str]:
        """
        Dada la serie temporal de interés, retorna el día de la semana
        con mayor promedio de interés por keyword.

        Returns:
            dict {keyword: nombre_dia_semana_en_español}
        """
        day_map = {
            0: 'Lunes', 1: 'Martes', 2: 'Miércoles',
            3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo',
        }
        result = {}
        for col in interest_df.columns:
            series = interest_df[col]
            if series.empty or series.sum() == 0:
                result[col] = '—'
                continue
            by_dow = series.groupby(series.index.dayofweek).mean()
            result[col] = day_map.get(int(by_dow.idxmax()), '—')
        return result
