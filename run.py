#!/usr/bin/env python
"""
Запуск системы:
- FastAPI сервер (порт 8000)
- Streamlit приложение (порт 8501)
"""
import subprocess
import sys
import os
import threading
import time
import webbrowser


def run_fastapi():
    """Запуск FastAPI сервера"""
    os.system("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


def run_streamlit():
    """Запуск Streamlit приложения"""
    time.sleep(2)  # Ждем FastAPI
    os.system("streamlit run app.py --server.port 8502")


def open_browser():
    """Открыть браузер с интерфейсом"""
    time.sleep(3)
    webbrowser.open("http://localhost:8502")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ЗАПУСК СИСТЕМЫ РЕКОМЕНДАЦИЙ ВАКАНСИЙ")
    print("=" * 60)
    print("📡 FastAPI API: http://localhost:8000")
    print("📄 API Docs: http://localhost:8000/docs")
    print("🎨 Streamlit UI: http://localhost:8502")
    print("=" * 60)

    # Запуск в отдельных потоках
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    browser_thread = threading.Thread(target=open_browser, daemon=True)

    fastapi_thread.start()
    streamlit_thread.start()
    browser_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Остановка серверов...")
        sys.exit(0)