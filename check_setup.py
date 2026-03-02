"""
Script de verificación de instalación
Verifica que todo esté configurado correctamente antes de ejecutar el agente
"""
import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Verifica si un archivo existe"""
    if Path(filepath).exists():
        print(f"  ✓ {description}: OK")
        return True
    else:
        print(f"  ✗ {description}: NO ENCONTRADO")
        return False


def check_env_variable(var_name, description):
    """Verifica si una variable de entorno está configurada"""
    value = os.getenv(var_name)
    if value and value.strip() and value != f'tu_{var_name.lower()}_aqui':
        print(f"  ✓ {description}: Configurado")
        return True
    else:
        print(f"  ✗ {description}: NO CONFIGURADO")
        return False


def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}: OK")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro}: Se requiere Python 3.8+")
        return False


def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    dependencies = [
        ('google-api-python-client', 'googleapiclient'),
        ('pandas', 'pandas'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('anthropic', 'anthropic'),
        ('python-dotenv', 'dotenv'),
    ]
    
    all_ok = True
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}: Instalado")
        except ImportError:
            print(f"  ✗ {package_name}: NO INSTALADO")
            all_ok = False
    
    return all_ok


def main():
    """Función principal de verificación"""
    print("=" * 70)
    print("VERIFICACIÓN DE INSTALACIÓN - YouTube AI Agent")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # 1. Verificar versión de Python
    print("1. Verificando versión de Python...")
    if not check_python_version():
        all_checks_passed = False
    print()
    
    # 2. Verificar archivos necesarios
    print("2. Verificando estructura de archivos...")
    files_to_check = [
        ('main.py', 'Script principal'),
        ('dashboard.py', 'Dashboard'),
        ('requirements.txt', 'Archivo de dependencias'),
        ('src/youtube_extractor.py', 'Extractor de YouTube'),
        ('src/database.py', 'Base de datos'),
        ('src/ai_analyzer.py', 'Analizador de IA'),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    print()
    
    # 3. Verificar dependencias
    print("3. Verificando dependencias de Python...")
    if not check_dependencies():
        all_checks_passed = False
        print("\n  ⚠️  Instala las dependencias con: pip install -r requirements.txt")
    print()
    
    # 4. Verificar archivo .env
    print("4. Verificando configuración (.env)...")
    
    # Cargar .env si existe
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("  ✓ Archivo .env encontrado")
        
        # Verificar variables
        env_checks = [
            ('YOUTUBE_API_KEY', 'YouTube API Key'),
            ('YOUTUBE_CHANNEL_IDS', 'Channel IDs'),
            ('ANTHROPIC_API_KEY', 'Anthropic API Key'),
        ]
        
        for var_name, description in env_checks:
            if not check_env_variable(var_name, description):
                all_checks_passed = False
    else:
        print("  ✗ Archivo .env NO ENCONTRADO")
        print("    Copia .env.example a .env y configura tus credenciales")
        all_checks_passed = False
    
    print()
    
    # 5. Verificar directorio de datos
    print("5. Verificando directorio de datos...")
    data_dir = Path('data')
    if data_dir.exists():
        print("  ✓ Directorio data/: OK")
    else:
        print("  ℹ️  Directorio data/: Se creará automáticamente")
    print()
    
    # Resumen final
    print("=" * 70)
    if all_checks_passed:
        print("✅ TODAS LAS VERIFICACIONES PASARON")
        print()
        print("Tu instalación está lista. Puedes ejecutar:")
        print("  python main.py          # Para extraer datos y generar recomendaciones")
        print("  streamlit run dashboard.py  # Para ver el dashboard")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print()
        print("Por favor revisa los errores arriba y:")
        print("  1. Instala las dependencias: pip install -r requirements.txt")
        print("  2. Copia .env.example a .env: cp .env.example .env")
        print("  3. Configura tus API keys en el archivo .env")
        print()
        print("Consulta el README.md para más detalles")
    print("=" * 70)
    print()
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
