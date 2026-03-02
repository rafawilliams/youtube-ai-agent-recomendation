"""
Módulo para extraer datos de YouTube usando la API v3
"""
import os
import logging
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from typing import List, Dict, Optional
import time

from retry_config import retry_google_api

log = logging.getLogger(__name__)


@retry_google_api
def _execute_with_retry(request):
    """Ejecuta una solicitud de la API de YouTube con retry en errores transitorios."""
    return request.execute()


class YouTubeDataExtractor:
    """Extrae datos de canales de YouTube usando la API oficial"""
    
    def __init__(self, api_key: str):
        """
        Inicializa el extractor con la API key
        
        Args:
            api_key: API key de Google Cloud
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def get_channel_info(self, channel_id: str) -> Dict:
        """
        Obtiene información básica del canal
        
        Args:
            channel_id: ID del canal de YouTube
            
        Returns:
            Diccionario con datos del canal
        """
        try:
            request = self.youtube.channels().list(
                part='snippet,statistics,contentDetails',
                id=channel_id
            )
            response = _execute_with_retry(request)
            
            if not response['items']:
                return None
                
            channel = response['items'][0]
            
            stats = channel['statistics']
            return {
                'channel_id': channel_id,
                'channel_name': channel['snippet']['title'],
                'description': channel['snippet']['description'],
                # subscriberCount se omite cuando el canal oculta su conteo
                'subscriber_count': int(stats.get('subscriberCount', 0)),
                'video_count': int(stats.get('videoCount', 0)),
                'view_count': int(stats.get('viewCount', 0)),
                'created_at': channel['snippet']['publishedAt'],
                'uploads_playlist_id': channel['contentDetails']['relatedPlaylists']['uploads']
            }
        except Exception as e:
            log.error("Error obteniendo info del canal: %s", e)
            return None
    
    def get_video_ids(self, channel_id: str, max_results: int = 50) -> List[str]:
        """
        Obtiene los IDs de los videos del canal
        
        Args:
            channel_id: ID del canal
            max_results: Cantidad máxima de videos a obtener
            
        Returns:
            Lista de IDs de videos
        """
        try:
            # Primero obtener el playlist ID de uploads
            channel_info = self.get_channel_info(channel_id)
            if not channel_info:
                return []
                
            playlist_id = channel_info['uploads_playlist_id']
            
            video_ids = []
            next_page_token = None
            
            while len(video_ids) < max_results:
                request = self.youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=next_page_token
                )
                response = _execute_with_retry(request)
                
                for item in response['items']:
                    video_ids.append(item['contentDetails']['videoId'])
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            return video_ids
            
        except HttpError as e:
            log.error("Error obteniendo IDs de videos: %s", e)
            return []
    
    def get_video_details(self, video_ids: List[str]) -> pd.DataFrame:
        """
        Obtiene detalles completos de los videos
        
        Args:
            video_ids: Lista de IDs de videos
            
        Returns:
            DataFrame con información detallada de videos
        """
        videos_data = []
        
        # La API permite máximo 50 videos por request
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            try:
                request = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails,status',
                    id=','.join(batch_ids)
                )
                response = _execute_with_retry(request)
                
                for video in response['items']:
                    # Determinar si es Short (duración < 60 segundos)
                    duration_str = video['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration_str)
                    is_short = duration_seconds <= 60
                    
                    video_data = {
                        'video_id': video['id'],
                        'title': video['snippet']['title'],
                        'description': video['snippet']['description'],
                        'published_at': video['snippet']['publishedAt'],
                        'channel_id': video['snippet']['channelId'],
                        'channel_title': video['snippet']['channelTitle'],
                        'duration': duration_str,
                        'duration_seconds': duration_seconds,
                        'is_short': is_short,
                        'video_type': 'Short' if is_short else 'Video Largo',
                        'view_count': int(video['statistics'].get('viewCount', 0)),
                        'like_count': int(video['statistics'].get('likeCount', 0)),
                        'comment_count': int(video['statistics'].get('commentCount', 0)),
                        'tags': ','.join(video['snippet'].get('tags', [])),
                        'category_id': video['snippet']['categoryId'],
                        'privacy_status': video['status']['privacyStatus'],
                    }
                    
                    # Calcular métricas derivadas
                    views = video_data['view_count']
                    if views > 0:
                        video_data['engagement_rate'] = (
                            (video_data['like_count'] + video_data['comment_count']) / views * 100
                        )
                    else:
                        video_data['engagement_rate'] = 0
                    
                    videos_data.append(video_data)
                    
                # Pequeña pausa para no exceder rate limits
                time.sleep(0.1)
                
            except HttpError as e:
                log.error("Error obteniendo detalles de videos: %s", e)
                continue
        
        return pd.DataFrame(videos_data)
    
    def get_video_analytics(self, video_id: str) -> Optional[Dict]:
        """
        Obtiene estadísticas adicionales de un video específico
        Nota: Algunas métricas requieren YouTube Analytics API
        
        Args:
            video_id: ID del video
            
        Returns:
            Diccionario con analytics adicionales
        """
        try:
            request = self.youtube.videos().list(
                part='statistics,contentDetails',
                id=video_id
            )
            response = _execute_with_retry(request)
            
            if not response['items']:
                return None
                
            stats = response['items'][0]['statistics']
            
            return {
                'video_id': video_id,
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'fetched_at': datetime.now().isoformat()
            }
            
        except HttpError as e:
            log.error("Error obteniendo analytics del video %s: %s", video_id, e)
            return None
    
    def _parse_duration(self, duration_str: str) -> int:
        """
        Convierte formato ISO 8601 de duración a segundos
        
        Args:
            duration_str: String en formato PT#H#M#S
            
        Returns:
            Duración en segundos
        """
        import re
        
        # Formato: PT1H2M3S
        hours = re.search(r'(\d+)H', duration_str)
        minutes = re.search(r'(\d+)M', duration_str)
        seconds = re.search(r'(\d+)S', duration_str)
        
        total_seconds = 0
        if hours:
            total_seconds += int(hours.group(1)) * 3600
        if minutes:
            total_seconds += int(minutes.group(1)) * 60
        if seconds:
            total_seconds += int(seconds.group(1))
            
        return total_seconds
    
    def extract_all_data(self, channel_ids: List[str], max_videos_per_channel: int = 50) -> pd.DataFrame:
        """
        Extrae todos los datos de múltiples canales
        
        Args:
            channel_ids: Lista de IDs de canales
            max_videos_per_channel: Máximo de videos por canal
            
        Returns:
            DataFrame consolidado con todos los videos
        """
        all_videos = []
        
        for channel_id in channel_ids:
            log.info("Extrayendo datos del canal: %s", channel_id)

            # Obtener info del canal
            channel_info = self.get_channel_info(channel_id)
            if not channel_info:
                log.warning("No se pudo obtener info del canal %s", channel_id)
                continue

            log.info("  Canal: %s", channel_info['channel_name'])
            log.info("  Suscriptores: %s", f"{channel_info['subscriber_count']:,}")

            # Obtener IDs de videos
            video_ids = self.get_video_ids(channel_id, max_videos_per_channel)
            log.info("  Videos encontrados: %d", len(video_ids))
            
            if not video_ids:
                continue
            
            # Obtener detalles de videos
            videos_df = self.get_video_details(video_ids)
            all_videos.append(videos_df)
            
            log.info("  Videos procesados: %d", len(videos_df))
        
        if not all_videos:
            return pd.DataFrame()
        
        # Consolidar todos los dataframes
        final_df = pd.concat(all_videos, ignore_index=True)
        
        # Convertir fechas
        final_df['published_at'] = pd.to_datetime(final_df['published_at'])
        final_df['days_since_published'] = (datetime.now(timezone.utc) - final_df['published_at']).dt.days
        
        return final_df


if __name__ == "__main__":
    # Test del extractor
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('YOUTUBE_API_KEY')
    channel_ids = os.getenv('YOUTUBE_CHANNEL_IDS', '').split(',')
    
    if not api_key:
        print("Error: YOUTUBE_API_KEY no configurada")
        exit(1)
    
    extractor = YouTubeDataExtractor(api_key)
    df = extractor.extract_all_data(channel_ids, max_videos_per_channel=20)
    
    print("\n=== RESUMEN DE EXTRACCIÓN ===")
    print(f"Total de videos: {len(df)}")
    print(f"\nPrimeros 5 videos:")
    print(df[['title', 'video_type', 'view_count', 'engagement_rate']].head())
