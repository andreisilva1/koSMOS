import os
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select
from checks.database import check_mongo_connection
from database.models import MLModel, SQLMLModel

load_dotenv()

MONGO_URL = os.getenv("DATABASE_URL")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

try:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[mongo_db]
    collection = db[mongo_collection]

except:
    client = None
    db = None
    collection = None

# For SQLite fallback
engine = create_async_engine(
    url="sqlite+aiosqlite:///dataset_local.sqlite", echo=False
)
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with async_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_model(self, ml_model: MLModel):
        # Verify MongoDB connection, with SQLite as a local fallback
        is_mongo_connected = await check_mongo_connection(client)
        if is_mongo_connected:
            data = ml_model.model_dump()
            result = await collection.insert_one(data)
            new_item_id = await collection.find_one({"_id": str(result.inserted_id)})
        else:
            new_item = SQLMLModel(id=uuid4(), **ml_model.model_dump())
            self.session.add(new_item)
            await self.session.commit()
            new_item_id = new_item.id
        return new_item_id

    async def load_model_from_db(self, model_id: str):
        is_mongo_connected = await check_mongo_connection(client)
        if is_mongo_connected:
            result = await collection.find_one({"name": model_id})
        else:
            query = await self.session.execute(
                select(SQLMLModel).where(SQLMLModel.name == model_id)
            )
            result_orm = query.scalar_one_or_none()
            result = result_orm.model_dump() if result_orm else None
        return result


def get_model_service(session: SessionDep):
    return ModelService(session)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


async def create_local_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
