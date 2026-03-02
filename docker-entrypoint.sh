#!/bin/bash
set -e

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-3306}"
DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-root}"

echo "Esperando MariaDB en $DB_HOST:$DB_PORT..."

for i in $(seq 1 30); do
    if python -c "
import pymysql, sys
try:
    pymysql.connect(host='$DB_HOST', port=int('$DB_PORT'),
                    user='$DB_USER', password='$DB_PASSWORD',
                    connect_timeout=2)
    print('OK')
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "MariaDB lista."
        exec "$@"
    fi
    echo "  Intento $i/30..."
    sleep 2
done

echo "ERROR: No se pudo conectar a MariaDB después de 60s"
exit 1
