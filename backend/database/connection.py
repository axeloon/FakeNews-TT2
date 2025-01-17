import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        logger.info("Creando nueva sesión de base de datos")
        yield db
    except Exception as e:
        logger.error(f"Error al crear sesión de base de datos: {e}")
        raise
    finally:
        logger.info("Cerrando sesión de base de datos")
        try:
            db.close()
        except Exception as e:
            logger.error(f"Error al cerrar sesión de base de datos: {e}")