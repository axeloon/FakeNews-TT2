from dotenv import load_dotenv
import os

load_dotenv()  # Carga las variables de entorno desde un archivo .env

X_EMAIL = os.getenv("X_EMAIL")
X_USERNAME = os.getenv("X_USERNAME")
X_PASSWORD = os.getenv("X_PASSWORD")

# Configuración de la base de datos
DATABASE_URL = "sqlite:///./backend/database/FakeNewsTT2.db"