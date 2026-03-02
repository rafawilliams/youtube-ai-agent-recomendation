FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para scikit-learn y PyMySQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios de runtime
RUN mkdir -p logs models config

# Configuración de Streamlit (tema oscuro, headless)
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Entrypoint: espera a que MariaDB esté lista
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]

EXPOSE 8501

# Default: ejecutar dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
