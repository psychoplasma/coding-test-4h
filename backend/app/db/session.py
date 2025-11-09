"""
Database session management
"""
import contextlib
from typing import Any, Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextlib.contextmanager
def scoped_session() -> Generator[Session, Any, None]:
    """Provide a session scope around a series of operations.

    Usage:
    ```python
    with session_scope() as session:
        session.get(...)
        session.add(...)
        # Commit will be called automatically if the block is exited without errors
        # Rollback will be called if an error occurs
    ```
    """
    with SessionLocal.begin() as session:
        yield session
