# ⚡ INICIO RÁPIDO - YouTube AI Agent

## 🚀 En 5 Minutos

### 1️⃣ Descargar e Instalar (2 min)

```bash
# Descomprimir el proyecto
unzip youtube-ai-agent.zip
cd youtube-ai-agent

# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Configurar Credenciales (2 min)

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar con tus credenciales
nano .env
```

**Necesitas:**
- ✅ YouTube API Key → [Obtener aquí](https://console.cloud.google.com/apis/credentials)
- ✅ Tu Channel ID → [Cómo obtenerlo](#obtener-channel-id)
- ✅ Anthropic API Key → [Obtener aquí](https://console.anthropic.com/)

### 3️⃣ Ejecutar (1 min)

```bash
# Verificar instalación
python check_setup.py

# Extraer datos y generar recomendación
python main.py

# Ver dashboard
streamlit run dashboard.py
```

¡Listo! 🎉

---

## 🔑 Obtener Channel ID

### Opción 1: Desde la URL
1. Ve a tu canal de YouTube
2. Mira la URL: `youtube.com/channel/UCxxxxxxxxxxxxxxxxx`
3. Copia la parte que empieza con `UC`

### Opción 2: Desde el código fuente
1. Ve a tu canal
2. Click derecho → "Ver código fuente"
3. Busca (Ctrl+F): `"channelId"`
4. Copia el valor entre comillas

### Opción 3: Desde YouTube Studio
1. Abre YouTube Studio
2. Settings → Channel → Advanced settings
3. Copia el "Channel ID"

---

## ⚙️ Configuración Mínima en .env

```bash
# YouTube API Key (REQUERIDO)
YOUTUBE_API_KEY=AIzaSyD_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Tu Channel ID (REQUERIDO)
YOUTUBE_CHANNEL_IDS=UCxxxxxxxxxxxxxxxxx

# Anthropic API Key (REQUERIDO)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 📋 Comandos Principales

```bash
# Verificar todo está OK
python check_setup.py

# Extraer datos y generar recomendación
python main.py

# Ver dashboard interactivo
streamlit run dashboard.py

# Ver ayuda
python main.py --help
```

---

## 🎯 Primera Ejecución - Qué Esperar

```
==============================================================
AGENTE DE IA PARA ANÁLISIS DE YOUTUBE
==============================================================

✓ Configuración cargada
  Canales a analizar: 1

PASO 1: Extrayendo datos de YouTube API...
------------------------------------------------------------
Extrayendo datos del canal: UCxxxxxxxxxxxxxxxxx
  Canal: Tu Canal
  Suscriptores: 1,234
  Videos encontrados: 45
  Videos procesados: 45

✓ Datos extraídos exitosamente
  Total de videos: 45
  Shorts: 28
  Videos Largos: 17

PASO 2: Guardando datos en base de datos...
------------------------------------------------------------
✓ Datos guardados en base de datos

PASO 3: Generando análisis con IA...
------------------------------------------------------------
📊 Analizando: Tu Canal

  📈 Estadísticas Clave:
     Total vistas: 98,765
     Promedio por video: 2,194
     Engagement rate: 4.35%

  🎬 Comparación:
     Shorts: 1,650 vistas promedio
     Videos Largos: 3,250 vistas promedio

  🎯 RECOMENDACIÓN PARA MAÑANA
  ========================================================
  
  📅 Fecha: 2024-02-18
  🎬 Formato: Video Largo
  📊 Performance esperado: ~3,250 vistas | ~4.5% engagement
  
  💡 Recomendación basada en que tus videos largos tienen
     mejor performance y engagement...

✓ PROCESO COMPLETADO EXITOSAMENTE
```

---

## 🐛 Problemas Comunes

### Error: "ModuleNotFoundError"
```bash
# Solución: Instalar dependencias
pip install -r requirements.txt
```

### Error: "YOUTUBE_API_KEY not configured"
```bash
# Solución: Verificar archivo .env
cat .env
# Asegurarse que tiene: YOUTUBE_API_KEY=tu_key_real
```

### Error: "Invalid API Key"
```bash
# Solución: Verificar en Google Cloud Console
# 1. API Key correcta
# 2. YouTube Data API v3 activada
# 3. Billing habilitado (si es necesario)
```

### Error: "No se pudieron extraer datos"
```bash
# Solución: Verificar Channel ID
# 1. Debe empezar con "UC"
# 2. Tu canal debe tener videos públicos
# 3. Channel ID debe ser correcto
```

---

## 💡 Próximos Pasos

Una vez que todo funcione:

1. **Explora el Dashboard**
   ```bash
   streamlit run dashboard.py
   ```
   - Ve tus métricas
   - Analiza tendencias
   - Revisa recomendaciones

2. **Ejecuta Diariamente**
   ```bash
   python main.py
   ```
   - Obtén nuevas recomendaciones
   - Actualiza tus datos
   - Sigue las sugerencias

3. **Lee la Documentación Completa**
   - `README.md` - Documentación completa
   - `USAGE_GUIDE.md` - Casos de uso prácticos

---

## 🆘 Necesitas Ayuda?

1. ✅ Ejecuta `python check_setup.py`
2. ✅ Lee los mensajes de error completos
3. ✅ Verifica que las credenciales son correctas
4. ✅ Consulta README.md para más detalles

---

## 📊 Estructura de Archivos

```
youtube-ai-agent/
├── main.py                    # ← Script principal
├── dashboard.py               # ← Dashboard Streamlit
├── check_setup.py            # ← Verificador de instalación
├── requirements.txt          # ← Dependencias
├── .env.example             # ← Plantilla de configuración
├── .env                     # ← Tu configuración (crear este)
├── README.md                # ← Documentación completa
├── USAGE_GUIDE.md          # ← Guía de uso detallada
├── QUICKSTART.md           # ← Este archivo
├── data/                   # ← Base de datos (se crea automáticamente)
└── src/                    # ← Código fuente
    ├── youtube_extractor.py
    ├── database.py
    └── ai_analyzer.py
```

---

## ⏱️ Tiempo Estimado

| Tarea | Tiempo |
|-------|--------|
| Instalación | 2-5 min |
| Obtener API Keys | 10-15 min |
| Configuración | 2 min |
| Primera ejecución | 2-5 min |
| **TOTAL** | **15-25 min** |

---

## ✅ Checklist de Verificación

Antes de ejecutar, asegúrate que tienes:

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Archivo `.env` creado
- [ ] YouTube API Key configurada
- [ ] Channel ID correcto
- [ ] Anthropic API Key configurada
- [ ] Al menos 5 videos públicos en tu canal

---

## 🎉 ¡Todo Listo!

Si llegaste hasta aquí y todo funciona, ¡felicitaciones! 

Ahora puedes:
- 📊 Ver tus métricas en el dashboard
- 🎯 Recibir recomendaciones diarias
- 📈 Optimizar tu estrategia de contenido
- 🚀 Hacer crecer tu canal

**¡Éxito con tu canal de YouTube!** 🎬
