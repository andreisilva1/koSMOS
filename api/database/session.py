from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

# For SQLite fallback
engine = create_async_engine(url="sqlite+aiosqlite:///local_db.sqlite", echo=False)
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with async_session() as session:
        yield session


async def create_local_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
