# 🤖 YouTube AI Agent - Agente de IA para Análisis de YouTube

Sistema inteligente que analiza tus métricas de YouTube y genera recomendaciones diarias sobre qué contenido publicar para maximizar audiencia y suscriptores.

## ✨ Características

- 📊 **Extracción automática de datos** de YouTube API
- 💾 **Base de datos SQLite** para almacenar histórico de métricas
- 🧠 **Análisis con IA** usando Claude de Anthropic
- 🎯 **Recomendaciones diarias** específicas (Shorts vs Videos Largos)
- 📈 **Dashboard interactivo** con Streamlit
- 📉 **Gráficos y visualizaciones** de performance
- 🔄 **Seguimiento de tendencias** a lo largo del tiempo

## 🎬 ¿Qué hace el agente?

El agente analiza:
- Vistas promedio de Shorts vs Videos Largos
- Engagement rate por tipo de contenido
- Tendencias de crecimiento
- Mejores días y horas para publicar
- Temas que han funcionado mejor
- Patrones de éxito en tu contenido

Y te recomienda:
- ✅ Si publicar un Short o Video Largo mañana
- ✅ Qué tipo de contenido crear
- ✅ Cuándo publicarlo
- ✅ Qué performance esperar

## 📋 Requisitos Previos

1. **Python 3.8 o superior**
2. **API Key de Google Cloud** (YouTube Data API v3)
3. **API Key de Anthropic** (Claude)
4. **Tus Channel IDs de YouTube**

## 🚀 Instalación

### Paso 1: Clonar o descargar el proyecto

```bash
# Si tienes el proyecto, navega al directorio
cd youtube-ai-agent
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Configurar credenciales

1. **Copia el archivo de ejemplo:**
```bash
cp .env.example .env
```

2. **Edita el archivo `.env` con tus credenciales:**

```bash
# YouTube API Key
YOUTUBE_API_KEY=tu_api_key_de_youtube

# IDs de tus canales (separados por coma si tienes varios)
YOUTUBE_CHANNEL_IDS=UCxxxxxxxxxxxxxxxxx

# Anthropic API Key
ANTHROPIC_API_KEY=tu_api_key_de_anthropic
```

### 📌 Cómo obtener las credenciales:

#### **YouTube API Key:**
1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Activa la "YouTube Data API v3"
4. Crea credenciales (API Key)
5. Copia el API Key generado

#### **YouTube Channel ID:**
1. Ve a tu canal de YouTube
2. Clic en tu avatar → "Ver tu canal"
3. En la URL verás algo como: `youtube.com/channel/UCxxxxxxxxxxxxxxxxx`
4. Copia el ID que empieza con "UC"

Alternativa:
1. Ve a tu canal
2. Clic derecho → "Ver código fuente"
3. Busca (Ctrl+F): "channelId"
4. Copia el valor

#### **Anthropic API Key:**
1. Ve a [Anthropic Console](https://console.anthropic.com/)
2. Crea una cuenta o inicia sesión
3. Ve a "API Keys"
4. Crea una nueva API Key
5. Copia el key generado

## 🎯 Uso

### Opción 1: Ejecutar el análisis completo (Recomendado)

```bash
python main.py
```

Este comando:
1. ✅ Extrae datos de tu(s) canal(es)
2. ✅ Los guarda en la base de datos
3. ✅ Genera análisis con IA
4. ✅ Crea recomendación para mañana

**Ejecútalo diariamente** para obtener recomendaciones actualizadas.

### Opción 2: Ver el Dashboard Interactivo

```bash
streamlit run dashboard.py
```

Abre tu navegador en `http://localhost:8501`

El dashboard incluye:
- 📊 Resumen general de métricas
- 📈 Análisis detallado de performance
- 🎯 Recomendaciones generadas
- 🤖 Generar nueva recomendación

## 📁 Estructura del Proyecto

```
youtube-ai-agent/
├── main.py                 # Script principal
├── dashboard.py            # Dashboard Streamlit
├── requirements.txt        # Dependencias
├── .env.example           # Plantilla de configuración
├── .env                   # Tu configuración (no subir a Git)
├── README.md              # Este archivo
├── data/                  # Datos y base de datos
│   └── youtube_analytics.db
├── src/                   # Código fuente
│   ├── youtube_extractor.py   # Extractor de YouTube API
│   ├── database.py            # Gestión de base de datos
│   └── ai_analyzer.py         # Motor de análisis con IA
└── config/                # Configuraciones adicionales
```

## 🔄 Workflow Recomendado

### Setup Inicial:
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar .env con tus credenciales
nano .env

# 3. Primera extracción de datos
python main.py

# 4. Ver resultados en dashboard
streamlit run dashboard.py
```

### Uso Diario:
```bash
# Cada día (o cuando quieras una nueva recomendación)
python main.py

# Ver dashboard actualizado
streamlit run dashboard.py
```

### Automatización (Opcional):

**Linux/Mac - Crontab:**
```bash
# Ejecutar todos los días a las 9 AM
crontab -e

# Agregar esta línea:
0 9 * * * cd /ruta/a/youtube-ai-agent && /usr/bin/python3 main.py
```

**Windows - Task Scheduler:**
1. Abre "Programador de tareas"
2. Crea una tarea básica
3. Programa para ejecutar diariamente
4. Acción: Iniciar programa
5. Programa: python.exe
6. Argumentos: main.py
7. Directorio: ruta a youtube-ai-agent

## 📊 Ejemplo de Salida

```
==============================================================
AGENTE DE IA PARA ANÁLISIS DE YOUTUBE
==============================================================

✓ Configuración cargada
  Canales a analizar: 1

PASO 1: Extrayendo datos de YouTube API...
------------------------------------------------------------
Extrayendo datos del canal: UCxxxxxxxxxxxxxxxxx
  Canal: Mi Canal de YouTube
  Suscriptores: 15,234
  Videos encontrados: 50
  Videos procesados: 50

✓ Datos extraídos exitosamente
  Total de videos: 50
  Shorts: 32
  Videos Largos: 18

PASO 2: Guardando datos en base de datos...
------------------------------------------------------------
✓ Datos guardados en base de datos
  Ubicación: data/youtube_analytics.db

PASO 3: Generando análisis con IA...
------------------------------------------------------------

📊 Analizando: Mi Canal de YouTube

  Generando análisis general...
  ✓ Análisis completado

  📈 Estadísticas Clave:
     Total vistas: 125,450
     Promedio por video: 2,509
     Engagement rate: 4.25%

  🎬 Comparación:
     Shorts: 1,850 vistas promedio
     Videos Largos: 3,750 vistas promedio

  🏆 Mejor video:
     Tutorial completo de Python para principiantes...
     12,450 vistas

  Generando recomendación para mañana...
  ✓ Recomendación generada

  ========================================================
  🎯 RECOMENDACIÓN PARA MAÑANA
  ========================================================

  📅 Fecha: 2024-02-18
  🎬 Formato: Video Largo
  📊 Performance esperado: ~3,750 vistas | ~4.5% engagement

  💡 Análisis completo:
  --------------------------------------------------------
  Basándome en tus datos, recomiendo crear un VIDEO LARGO
  mañana porque:

  1. PERFORMANCE SUPERIOR: Tus videos largos obtienen 2x más
     vistas que los shorts (3,750 vs 1,850)
  
  2. MEJOR ENGAGEMENT: Los videos largos tienen engagement
     del 4.8% vs 3.7% de los shorts
  
  3. MOMENTO IDEAL: Publicar los martes a las 18:00 ha
     mostrado el mejor rendimiento histórico
  
  4. TEMA RECOMENDADO: Crea contenido tipo tutorial o
     explicativo similar a tu mejor video, que obtuvo
     12,450 vistas...

==============================================================
✓ PROCESO COMPLETADO EXITOSAMENTE
==============================================================
```

## 🛠️ Solución de Problemas

### Error: "YOUTUBE_API_KEY no configurada"
- Verifica que el archivo `.env` existe
- Asegúrate de que la API Key está correcta
- No dejes espacios antes o después del `=`

### Error: "Invalid API Key"
- Verifica que activaste YouTube Data API v3 en Google Cloud
- Asegúrate de que el proyecto tiene billing habilitado
- Regenera el API Key si es necesario

### Error: "No se pudieron extraer datos"
- Verifica que tu Channel ID es correcto (empieza con UC)
- Asegúrate de tener al menos 1 video público
- Verifica que tu canal no es privado

### Error de Anthropic API
- Verifica tu API Key de Anthropic
- Revisa que tienes créditos disponibles
- Asegúrate de tener acceso a Claude Sonnet

## 📈 Mejoras Futuras

- [ ] Integración con YouTube Analytics API para métricas más detalladas
- [ ] Análisis de thumbnails con visión por computadora
- [x] Sugerencias de títulos optimizados con IA (3 opciones con análisis de palabras clave, claridad y curiosidad)
- [ ] Análisis de competencia
- [ ] Predicciones de viralidad
- [ ] Notificaciones automáticas
- [ ] App móvil

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el agente:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Puedes usarlo libremente para tus canales.

## 🆘 Soporte

Si tienes problemas o preguntas:

1. Revisa la sección "Solución de Problemas"
2. Verifica que todas las credenciales están correctas
3. Asegúrate de tener Python 3.8+
4. Revisa los logs de error completos

## 🙏 Créditos

- **YouTube Data API** para acceso a métricas
- **Anthropic Claude** para análisis inteligente
- **Streamlit** para el dashboard interactivo
- **Plotly** para visualizaciones

---

**Desarrollado con ❤️ para creadores de contenido de YouTube**

*¿Te gusta el proyecto? Dale una ⭐ en GitHub!*
