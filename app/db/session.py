# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from app.core.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DB_PORT
# from pgvector.psycopg2 import register_vector


# engine = create_engine(
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
#     pool_pre_ping=True,
# )

# # Register pgvector with psycopg2
# with engine.connect() as conn:
#     raw = conn.connection
#     register_vector(raw)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app.core.config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DB_PORT
from pgvector.psycopg2 import register_vector


DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# Register pgvector on EVERY new connection
@event.listens_for(engine, "connect")
def register_pgvector(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)