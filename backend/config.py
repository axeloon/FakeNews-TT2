from dotenv import load_dotenv
import os

load_dotenv()  # Carga las variables de entorno desde un archivo .env

X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")