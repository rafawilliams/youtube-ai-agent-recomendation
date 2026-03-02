"""
Exportación a Google Calendar y Google Sheets.

Usa un token OAuth 2.0 separado (token_integrations.json) para no
interferir con el token de YouTube Analytics (token.json).

Requisitos previos:
  1. En Google Cloud Console (mismo proyecto de credentials.json):
     - Habilitar "Google Calendar API"
     - Habilitar "Google Sheets API"
  2. credentials.json ya debe existir en la raíz del proyecto.
  3. La primera vez que se use, se abrirá el navegador para autorizar.
     El token se persiste en token_integrations.json.
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from retry_config import retry_google_api

log = logging.getLogger(__name__)


@retry_google_api
def _execute_with_retry(request):
    """Ejecuta una solicitud de Google API con retry en errores transitorios."""
    return request.execute()

INTEGRATION_SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets',
]

# Google Calendar color IDs (numéricas)
# Ref: https://developers.google.com/calendar/api/v3/reference/colors
CALENDAR_COLOR_VIDEO_LARGO = '9'  # Bold Blue
CALENDAR_COLOR_SHORT = '6'       # Tangerine


def get_integration_credentials():
    """
    Obtiene credenciales OAuth para Calendar y Sheets.
    Usa token_integrations.json separado del token de Analytics.

    Returns:
        google.oauth2.credentials.Credentials

    Raises:
        FileNotFoundError: si credentials.json no existe.
    """
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds_file = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
    token_file = os.getenv('GOOGLE_INTEGRATIONS_TOKEN_FILE', 'token_integrations.json')

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, INTEGRATION_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_file):
                raise FileNotFoundError(
                    f"'{creds_file}' no encontrado. "
                    "Descárgalo desde Google Cloud Console → Credenciales → "
                    "OAuth 2.0 (Aplicación de escritorio)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, INTEGRATION_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, 'w') as f:
            f.write(creds.to_json())

    return creds


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar
# ─────────────────────────────────────────────────────────────────────────────

class GoogleCalendarExporter:
    """Exporta el plan semanal como eventos de Google Calendar."""

    def __init__(self):
        self._service = None

    def _get_service(self):
        if not self._service:
            from googleapiclient.discovery import build
            creds = get_integration_credentials()
            self._service = build('calendar', 'v3', credentials=creds)
        return self._service

    def export_weekly_plan(
        self,
        days_data: List[Dict],
        strategy: str = '',
        calendar_id: str = 'primary',
    ) -> Dict:
        """
        Exporta los días con publish=True como eventos de Google Calendar.

        Args:
            days_data: Lista de dicts del plan semanal.
            strategy: Texto de estrategia general de la semana.
            calendar_id: ID del calendario destino ('primary' = principal).

        Returns:
            {created, skipped, errors, event_links}
        """
        service = self._get_service()
        result = {'created': 0, 'skipped': 0, 'errors': [], 'event_links': []}

        for day in days_data:
            if not day.get('publish'):
                continue

            try:
                if self._event_exists(service, calendar_id, day):
                    result['skipped'] += 1
                    continue

                event = self._build_event(day, strategy)
                request = service.events().insert(
                    calendarId=calendar_id,
                    body=event,
                )
                created = _execute_with_retry(request)

                result['created'] += 1
                result['event_links'].append(created.get('htmlLink', ''))

            except Exception as e:
                result['errors'].append(f"{day.get('date', '?')}: {e}")

        return result

    def _build_event(self, day: Dict, strategy: str) -> Dict:
        """Construye el dict del evento para la Calendar API."""
        vtype = day.get('type', '')
        topic = day.get('topic', '')
        hour = day.get('hour', 12)
        date_str = day.get('date', '')
        reason = day.get('reason', '')

        emoji = '🎬' if vtype == 'Video Largo' else '📱'
        summary = f"{emoji} {vtype}: {topic}"

        description_parts = [f"Tipo: {vtype}", f"Tema: {topic}"]
        if reason:
            description_parts.append(f"\nRazón: {reason}")
        if strategy:
            description_parts.append(f"\nEstrategia semanal:\n{strategy}")
        description_parts.append("\n— Generado por YouTube AI Agent")
        description = "\n".join(description_parts)

        end_hour = hour + 1 if hour < 23 else 23
        end_minute = 59 if hour >= 23 else 0

        color_id = CALENDAR_COLOR_VIDEO_LARGO if vtype == 'Video Largo' else CALENDAR_COLOR_SHORT

        return {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': f"{date_str}T{hour:02d}:00:00",
                'timeZone': 'America/Panama',
            },
            'end': {
                'dateTime': f"{date_str}T{end_hour:02d}:{end_minute:02d}:00",
                'timeZone': 'America/Panama',
            },
            'colorId': color_id,
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 120},
                ],
            },
        }

    def _event_exists(self, service, calendar_id: str, day: Dict) -> bool:
        """Verifica si ya existe un evento con el mismo título en la fecha dada."""
        date_str = day.get('date', '')
        if not date_str:
            return False

        vtype = day.get('type', '')
        topic = day.get('topic', '')
        emoji = '🎬' if vtype == 'Video Largo' else '📱'
        expected_summary = f"{emoji} {vtype}: {topic}"

        try:
            request = service.events().list(
                calendarId=calendar_id,
                timeMin=f"{date_str}T00:00:00-05:00",
                timeMax=f"{date_str}T23:59:59-05:00",
                singleEvents=True,
                q=topic[:30],
            )
            events_result = _execute_with_retry(request)

            for event in events_result.get('items', []):
                if event.get('summary', '') == expected_summary:
                    return True
        except Exception as e:
            log.debug("Error verificando evento existente: %s", e)

        return False


# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets
# ─────────────────────────────────────────────────────────────────────────────

class GoogleSheetsExporter:
    """Exporta datos del canal a Google Sheets."""

    def __init__(self):
        self._service = None

    def _get_service(self):
        if not self._service:
            from googleapiclient.discovery import build
            creds = get_integration_credentials()
            self._service = build('sheets', 'v4', credentials=creds)
        return self._service

    def export_weekly_plan(
        self,
        days_data: List[Dict],
        strategy: str = '',
        week_start_date: str = '',
        channel_name: str = '',
    ) -> Dict:
        """
        Crea un Google Sheet con el plan semanal.

        Returns:
            {spreadsheet_id, spreadsheet_url}
        """
        service = self._get_service()

        title = f"Plan Semanal - {channel_name} - {week_start_date}"

        request = service.spreadsheets().create(
            body={
                'properties': {'title': title},
                'sheets': [{'properties': {'title': 'Plan Semanal'}}],
            }
        )
        spreadsheet = _execute_with_retry(request)

        spreadsheet_id = spreadsheet['spreadsheetId']
        spreadsheet_url = spreadsheet['spreadsheetUrl']

        headers = ['Día', 'Fecha', 'Publicar', 'Formato', 'Tema', 'Hora', 'Razón']
        rows = [headers]
        for day in days_data:
            rows.append([
                day.get('day', ''),
                day.get('date', ''),
                'Sí' if day.get('publish') else 'No',
                day.get('type', '') or '—',
                day.get('topic', '') or '—',
                f"{day['hour']}:00" if day.get('hour') is not None else '—',
                day.get('reason', '') or '—',
            ])

        rows.append([])
        rows.append(['Estrategia:', strategy])

        request = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Plan Semanal!A1',
            valueInputOption='RAW',
            body={'values': rows},
        )
        _execute_with_retry(request)

        self._format_header(service, spreadsheet_id, sheet_id=0, num_cols=len(headers))

        return {'spreadsheet_id': spreadsheet_id, 'spreadsheet_url': spreadsheet_url}

    def export_video_metrics(
        self,
        videos_df: pd.DataFrame,
        channel_name: str = '',
    ) -> Dict:
        """
        Crea un Google Sheet con métricas de todos los videos.

        Returns:
            {spreadsheet_id, spreadsheet_url}
        """
        service = self._get_service()

        title = f"Métricas - {channel_name} - {datetime.now().strftime('%Y-%m-%d')}"

        request = service.spreadsheets().create(
            body={
                'properties': {'title': title},
                'sheets': [{'properties': {'title': 'Videos'}}],
            }
        )
        spreadsheet = _execute_with_retry(request)

        spreadsheet_id = spreadsheet['spreadsheetId']
        spreadsheet_url = spreadsheet['spreadsheetUrl']

        export_cols = [
            ('title', 'Título'),
            ('video_type', 'Tipo'),
            ('published_at', 'Fecha publicación'),
            ('view_count', 'Vistas'),
            ('like_count', 'Likes'),
            ('comment_count', 'Comentarios'),
            ('engagement_rate', 'Engagement %'),
            ('duration_seconds', 'Duración (s)'),
        ]

        headers = [display for _, display in export_cols]
        rows = [headers]

        for _, video in videos_df.iterrows():
            row = []
            for col_key, _ in export_cols:
                val = video.get(col_key, '')
                if col_key == 'published_at' and pd.notna(val):
                    val = str(val)[:16]
                elif col_key == 'engagement_rate' and pd.notna(val):
                    val = round(float(val), 2)
                elif pd.isna(val):
                    val = ''
                else:
                    val = val.item() if hasattr(val, 'item') else val
                row.append(val)
            rows.append(row)

        request = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Videos!A1',
            valueInputOption='RAW',
            body={'values': rows},
        )
        _execute_with_retry(request)

        self._format_header(service, spreadsheet_id, sheet_id=0, num_cols=len(headers))

        return {'spreadsheet_id': spreadsheet_id, 'spreadsheet_url': spreadsheet_url}

    def _format_header(self, service, spreadsheet_id: str, sheet_id: int, num_cols: int):
        """Aplica formato al header: negrita, fondo oscuro, texto blanco, fila congelada."""
        try:
            request = service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    'requests': [
                        {
                            'repeatCell': {
                                'range': {
                                    'sheetId': sheet_id,
                                    'startRowIndex': 0,
                                    'endRowIndex': 1,
                                    'startColumnIndex': 0,
                                    'endColumnIndex': num_cols,
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'backgroundColor': {
                                            'red': 0.06, 'green': 0.09, 'blue': 0.16,
                                        },
                                        'textFormat': {
                                            'bold': True,
                                            'foregroundColor': {
                                                'red': 0.97, 'green': 0.98, 'blue': 0.99,
                                            },
                                        },
                                    }
                                },
                                'fields': 'userEnteredFormat(backgroundColor,textFormat)',
                            }
                        },
                        {
                            'updateSheetProperties': {
                                'properties': {
                                    'sheetId': sheet_id,
                                    'gridProperties': {'frozenRowCount': 1},
                                },
                                'fields': 'gridProperties.frozenRowCount',
                            }
                        },
                    ]
                },
            )
            _execute_with_retry(request)
        except Exception as e:
            log.debug("Error aplicando formato al header de Sheets: %s", e)
