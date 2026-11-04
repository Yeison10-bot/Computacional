"""Verifica que el entorno de desarrollo está listo.

Ejecutar desde la raíz del proyecto:
    .venv\\Scripts\\python.exe vision\\verificar_entorno.py

Comprueba cada biblioteca por separado e informa la versión instalada, de modo
que un fallo en una no oculte el estado de las demás.
"""

import importlib
import sys

BIBLIOTECAS = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("ultralytics", "Ultralytics (YOLO)"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("sqlalchemy", "SQLAlchemy"),
    ("psycopg", "Psycopg (PostgreSQL)"),
    ("dotenv", "python-dotenv"),
    ("requests", "Requests"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
]


def verificar_bibliotecas():
    faltantes = []
    for modulo, nombre in BIBLIOTECAS:
        try:
            m = importlib.import_module(modulo)
            print(f"  [OK]  {nombre:<22} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  [X]   {nombre:<22} NO instalado")
            faltantes.append(nombre)
    return faltantes


def verificar_gpu():
    import torch

    if not torch.cuda.is_available():
        print("  [X]   CUDA no disponible: el entrenamiento correría en CPU.")
        return False
    print(f"  [OK]  GPU: {torch.cuda.get_device_name(0)}")
    print(f"        CUDA de PyTorch: {torch.version.cuda}")
    memoria = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"        Memoria: {memoria:.1f} GB")
    # Una operación real en la GPU confirma que funciona, no solo que se detecta.
    x = torch.rand(1000, 1000, device="cuda")
    print(f"        Prueba de cálculo en GPU: OK (suma = {(x @ x).sum().item():.0f})")
    return True


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} en {sys.executable}\n")
    print("Bibliotecas:")
    faltantes = verificar_bibliotecas()
    print("\nGPU:")
    gpu_ok = "PyTorch" not in faltantes and verificar_gpu()
    print()
    if faltantes or not gpu_ok:
        print("Entorno INCOMPLETO.")
        sys.exit(1)
    print("Entorno listo.")
