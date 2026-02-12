from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DB_PORT

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
