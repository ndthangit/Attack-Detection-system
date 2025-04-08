from sqlmodel import create_engine
from core.config import database_settings

DATABASE_URL = f"postgresql://{database_settings.DB_USER}:{database_settings.DB_PASSWORD}@{database_settings.DB_HOST}:{database_settings.DB_PORT}/{database_settings.DB_NAME}"

engine = create_engine(DATABASE_URL, echo=True)

if not engine:
    raise ConnectionError("Database engine not initialized")