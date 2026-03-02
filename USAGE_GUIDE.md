# 📚 Guía de Uso - Ejemplos Prácticos

Esta guía te muestra cómo usar el YouTube AI Agent en diferentes escenarios.

## 🎯 Caso de Uso 1: Primera Vez - Setup Inicial

### Situación
Acabas de descargar el proyecto y quieres empezar a usar el agente.

### Pasos:

```bash
# 1. Verificar que todo está instalado correctamente
python check_setup.py

# 2. Si faltan dependencias, instalarlas
pip install -r requirements.txt

# 3. Configurar credenciales
cp .env.example .env
nano .env  # o usa tu editor favorito

# 4. Ejecutar primera extracción
python main.py

# 5. Ver resultados en dashboard
streamlit run dashboard.py
```

### Resultado Esperado:
- Base de datos creada con tus videos
- Primera recomendación generada
- Dashboard mostrando tus métricas

---

## 🎯 Caso de Uso 2: Revisión Diaria de Recomendaciones

### Situación
Ya tienes el sistema configurado y quieres ver qué publicar hoy.

### Flujo de Trabajo:

```bash
# Opción A: Desde línea de comandos
python main.py

# Opción B: Desde el dashboard
streamlit run dashboard.py
# Ir a: "🎯 Recomendaciones"
```

### Qué Esperar:
```
🎯 RECOMENDACIÓN PARA MAÑANA
========================================================

📅 Fecha: 2024-02-18
🎬 Formato: Short
📊 Performance esperado: ~2,500 vistas | ~5.2% engagement

💡 Análisis completo:
--------------------------------------------------------
Recomiendo crear un SHORT mañana por las siguientes razones:

1. TENDENCIA RECIENTE: Tus últimos 5 shorts han superado
   el promedio del canal en un 35%

2. ENGAGEMENT: Los shorts están generando 1.8x más
   engagement que videos largos este mes

3. MOMENTO: Los martes históricamente tienen el mejor
   CTR para shorts en tu canal

4. TEMA SUGERIDO: Crea contenido rápido estilo
   "Top 3 consejos" o "Error común" similar a tu
   short de hace 2 semanas que obtuvo 8,450 vistas
```

---

## 🎯 Caso de Uso 3: Análisis Profundo de Performance

### Situación
Quieres entender qué está funcionando y qué no en tu canal.

### Cómo Hacerlo:

```bash
# Abrir dashboard
streamlit run dashboard.py

# Navegar a: "📈 Análisis de Performance"
```

### Insights que Obtendrás:

1. **Timeline de Publicaciones**
   - Visualización de todos tus videos en el tiempo
   - Vistas por fecha de publicación
   - Identificación de picos y valles

2. **Top 10 Videos**
   - Cuáles han tenido mejor performance
   - Qué tipo de contenido (Short/Largo)
   - Engagement de cada uno

3. **Comparación Shorts vs Videos Largos**
   - Promedio de vistas por tipo
   - Engagement rate comparado
   - Distribución de performance

### Decisiones que Puedes Tomar:
- "Mis shorts están teniendo mejor engagement → Hacer más shorts"
- "Videos largos el jueves funcionan mejor → Programar para jueves"
- "Tutorial tipo X tuvo 10k vistas → Hacer más de ese estilo"

---

## 🎯 Caso de Uso 4: Planificación Semanal

### Situación
Quieres planificar tu contenido para toda la semana.

### Estrategia:

```bash
# Lunes: Generar recomendación
python main.py

# Martes: Revisar análisis
streamlit run dashboard.py

# Miércoles: Generar nueva recomendación (si quieres actualizar)
# Ir al dashboard → "🤖 Generar Nueva Recomendación"
```

### Ejemplo de Plan Semanal Basado en Datos:

**Lunes:**
- Recomendación: Video Largo - Tutorial
- Razón: Mejor engagement en lunes
- Tema: Similar al mejor video del mes

**Miércoles:**
- Recomendación: Short
- Razón: Mitad de semana tiene alta viralidad
- Tema: Tips rápidos

**Viernes:**
- Recomendación: Video Largo
- Razón: Audiencia tiene más tiempo el fin de semana
- Tema: Contenido entretenimiento

---

## 🎯 Caso de Uso 5: Comparar Múltiples Canales

### Situación
Tienes varios canales y quieres analizar todos.

### Configuración:

```bash
# Editar .env
YOUTUBE_CHANNEL_IDS=UCxxxxxxxx1,UCxxxxxxxx2,UCxxxxxxxx3
```

### Ejecutar:

```bash
python main.py
```

### Resultado:
El agente procesará cada canal por separado:
```
📊 Analizando: Mi Canal Gaming
  ✓ Análisis completado
  🎯 Recomendación: Short sobre nuevo juego

📊 Analizando: Mi Canal Educativo
  ✓ Análisis completado
  🎯 Recomendación: Video Largo tipo tutorial

📊 Analizando: Mi Canal Vlogs
  ✓ Análisis completado
  🎯 Recomendación: Short de día en mi vida
```

---

## 🎯 Caso de Uso 6: Debugging y Troubleshooting

### Situación
Algo no funciona o quieres ver más detalles.

### Herramientas:

```bash
# 1. Verificar setup
python check_setup.py

# 2. Ver datos en la base de datos
python
>>> from src.database import YouTubeDatabase
>>> db = YouTubeDatabase()
>>> df = db.get_all_videos()
>>> print(df.head())
>>> print(f"Total videos: {len(df)}")

# 3. Probar extractor manualmente
python
>>> from src.youtube_extractor import YouTubeDataExtractor
>>> import os
>>> from dotenv import load_dotenv
>>> load_dotenv()
>>> extractor = YouTubeDataExtractor(os.getenv('YOUTUBE_API_KEY'))
>>> info = extractor.get_channel_info('TU_CHANNEL_ID')
>>> print(info)
```

---

## 🎯 Caso de Uso 7: Automatización Completa

### Situación
Quieres que el sistema se ejecute automáticamente todos los días.

### Linux/Mac con Crontab:

```bash
# Editar crontab
crontab -e

# Agregar línea (ejecutar diariamente a las 8 AM)
0 8 * * * cd /home/usuario/youtube-ai-agent && /usr/bin/python3 main.py >> logs/cron.log 2>&1

# Crear directorio de logs
mkdir logs
```

### Windows con Task Scheduler:

1. Abre "Programador de tareas"
2. "Crear tarea básica"
3. Nombre: "YouTube AI Agent"
4. Desencadenador: Diariamente a las 8:00 AM
5. Acción: Iniciar programa
   - Programa: `C:\Python39\python.exe`
   - Argumentos: `main.py`
   - Iniciar en: `C:\ruta\a\youtube-ai-agent`

### Python Script de Automatización:

```python
# auto_run.py
import schedule
import time
import subprocess
from datetime import datetime

def run_agent():
    print(f"\n[{datetime.now()}] Ejecutando YouTube AI Agent...")
    try:
        result = subprocess.run(['python', 'main.py'], 
                              capture_output=True, 
                              text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error ejecutando agente: {e}")

# Programar ejecución diaria a las 8 AM
schedule.every().day.at("08:00").do(run_agent)

print("Agente programado para ejecutarse diariamente a las 8 AM")
print("Presiona Ctrl+C para detener...")

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🎯 Caso de Uso 8: Integración con tu Workflow

### Situación
Quieres integrar las recomendaciones en tu proceso de creación.

### Workflow Sugerido:

```
1. LUNES MAÑANA (Planificación)
   ├─ Ejecutar: python main.py
   ├─ Revisar dashboard
   └─ Planificar semana

2. LUNES-VIERNES (Creación)
   ├─ Crear contenido según recomendación
   ├─ Grabar/Editar
   └─ Preparar para publicación

3. MOMENTO DE PUBLICAR
   ├─ Revisar "mejor hora" del análisis
   ├─ Publicar según recomendación
   └─ Usar insights para título/descripción

4. DESPUÉS DE PUBLICAR
   ├─ Esperar 24-48 horas
   ├─ Ejecutar main.py de nuevo
   └─ Ver cómo performó vs predicción

5. FIN DE SEMANA (Análisis)
   ├─ Revisar dashboard completo
   ├─ Analizar tendencias
   └─ Ajustar estrategia
```

---

## 💡 Tips y Mejores Prácticas

### 1. Frecuencia de Actualización
- **Mínimo**: 1 vez por semana
- **Óptimo**: 2-3 veces por semana
- **Máximo**: Diariamente (antes de crear contenido)

### 2. Cantidad de Datos
- **Mínimo**: 10 videos para empezar
- **Óptimo**: 30+ videos para mejores insights
- **Ideal**: 50+ videos con métricas históricas

### 3. Interpretación de Recomendaciones
- ✅ Usa las recomendaciones como GUÍA
- ✅ Combina con tu intuición creativa
- ✅ Experimenta y prueba
- ❌ No las sigas ciegamente

### 4. Maximizar Precisión
- Ejecuta el agente después de publicar (para capturar nuevo video)
- Deja que los videos "maduren" (48-72 horas)
- Analiza patrones a largo plazo, no solo videos individuales

### 5. Cuando NO Seguir la Recomendación
- Si el tema sugerido no se alinea con tu marca
- Si tienes contenido urgente/trending que publicar
- Si la recomendación contradice eventos actuales
- Si tienes contenido patrocinado programado

---

## 🔧 Personalización Avanzada

### Modificar Cantidad de Videos Analizados

```python
# En main.py, línea ~50
videos_df = extractor.extract_all_data(
    channel_ids, 
    max_videos_per_channel=100  # Cambiar de 50 a 100
)
```

### Cambiar Período de Análisis

```python
# En .env
DAYS_TO_ANALYZE=60  # Analizar últimos 60 días en lugar de 90
```

### Personalizar Prompts de IA

```python
# En src/ai_analyzer.py
# Modificar función _create_recommendation_prompt()
# Agrega tus criterios específicos
```

---

## 📊 Métricas Clave a Vigilar

### Indicadores de Éxito:
1. **Crecimiento de suscriptores**: +5-10% mensual
2. **Vistas promedio**: Tendencia ascendente
3. **Engagement rate**: >3% es bueno, >5% es excelente
4. **CTR**: >10% es bueno para thumbnails/títulos

### Red Flags:
- ⚠️ Vistas cayendo consistentemente
- ⚠️ Engagement rate <1%
- ⚠️ Shorts con 0 comentarios
- ⚠️ Videos con más dislikes que likes

---

## 🎓 Aprendizaje Continuo

El agente mejora mientras más lo usas:
1. Más datos históricos = Mejores predicciones
2. Más recomendaciones seguidas = Mejor calibración
3. Feedback continuo = Insights más precisos

### Llevar un Log Manual:
```
Fecha: 2024-02-18
Recomendación: Short sobre Python
Seguí recomendación: Sí
Resultado Real: 3,200 vistas (predicción: 2,500)
Nota: Funcionó mejor de lo esperado
```

---

¿Necesitas ayuda con algún caso de uso específico? ¡Consulta el README.md o abre un issue!
