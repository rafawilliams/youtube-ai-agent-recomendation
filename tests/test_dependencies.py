"""
Compatibilidad real de dependencias: el resto de tests simula los clientes,
así que una combinación de versiones rota pasaría desapercibida sin esto.
No hace llamadas de red: solo construye los objetos.
"""


def test_anthropic_client_builds_with_installed_httpx():
    # anthropic 0.18.x + httpx 0.28 lanzaba TypeError ('proxies') al construir
    from anthropic import Anthropic

    client = Anthropic(api_key="test-key")

    assert client.messages is not None


def test_engine_dependencies_importable():
    # Motor local de la app de escritorio (T002)
    import fastapi
    import httpx
    import uvicorn

    assert fastapi.FastAPI
    assert uvicorn.run
    assert httpx.Client
