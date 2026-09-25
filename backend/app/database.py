import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DB_HOST=os.getenv("DB_HOST", "localhost")
DB_PORT=os.getenv("DB_PORT", "5433")
DB_NAME=os.getenv("DB_NAME", "scamshield_db")
DB_USER=os.getenv("DB_USER", "postgres")
DB_PASSWORD=os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    if DB_PASSWORD
    else None
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = (
    sessionmaker(bind=engine, autoflush=False, autocommit=False)
    if engine
    else None
)

class Base(DeclarativeBase):
    pass

def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database is not configured. Set DB_PASSWORD in backend/.env")
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_connection():
    if engine is None:
        raise RuntimeError("Database is not configured. Set DB_PASSWORD in backend/.env")
    with engine.connect() as connection:
        return connection.execute(text("SELECT current_database(), current_user, inet_server_port()" )).fetchone()
