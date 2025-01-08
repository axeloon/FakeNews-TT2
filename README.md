# FakeNews-TT2

Sistema de detección de noticias falsas utilizando técnicas de Machine Learning y procesamiento de lenguaje natural.

## Descripción del Proyecto
Este proyecto implementa una API REST para el entrenamiento y evaluación de modelos de detección de noticias falsas, utilizando múltiples modelos de Machine Learning y análisis de sentimientos. El sistema puede procesar contenido de redes sociales y sitios de noticias.

## Stack Tecnológico
### Backend
- Python 3.10+
- FastAPI
- Uvicorn
- SQLAlchemy
- Poetry
- TensorFlow
- Scikit-learn
- Pandas
- BeautifulSoup4
- Playwright

### Base de Datos
- SQLite

## Requisitos Previos
- Python 3.10 o superior
- Poetry (gestor de dependencias)

### Instalación de Poetry

Opcion 1: instalar poetry desde el instalador oficial
1. Descargar Poetry desde [https://python-poetry.org/](https://python-poetry.org/)
2. Ejecutar el instalador
3. Agregar Poetry a la ruta del sistema

Opcion 2: instalar poetry desde pip

1. Instalar Poetry en Windows
   ```bash
   pip install poetry
   ```

## Instalación del Proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/axeloon/FakeNews-TT2.git
   ```

2. Navegar al directorio del proyecto:
   ```bash
   cd FakeNews-TT2
   ```

3. Instalar las dependencias del proyecto con Poetry:
   ```bash
   poetry install
   ```

6. Instalar las dependencias de Playwright:
   ```bash
   poetry run playwright install
   ```

## Configuración

1. Crear un archivo `.env` en la raíz del proyecto con las variables necesarias:
   ```env
    X_EMAIL="usuario.prueba@example.com"
    X_USERNAME="UsuarioPrueba123"
    X_PASSWORD="ContraseñaSegura2024"
   ```

## Ejecución

1. Iniciar el servidor de desarrollo:
   ```bash
   poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info
   ```

2. Acceder a la API en:
   [http://localhost:8000](http://localhost:8000)

3. Consultar la documentación interactiva en:
   [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

## Rutas API

### Procesamiento de Datos (/api)
El sistema cuenta con diferentes routers para el procesamiento de datos:

- **CSV Processing**: Permite procesar archivos CSV, realizar análisis estadísticos, generar visualizaciones y aplicar análisis de sentimientos a los datos.

- **Twitter/X**: Facilita la extracción y procesamiento de tweets, permitiendo buscar usuarios específicos o procesar listas predefinidas de usuarios.

- **Entrenamiento de Modelos**: Proporciona endpoints para entrenar y ajustar modelos de machine learning, con y sin análisis de sentimientos, incluyendo capacidades de fine-tuning.

- **EMOL**: Permite la extracción y procesamiento de noticias históricas del portal EMOL, con capacidad de scraping y almacenamiento automático.

- **Noticias**: Gestiona el acceso a la base de datos de noticias recolectadas, permitiendo consultas paginadas y filtradas de la información almacenada.

