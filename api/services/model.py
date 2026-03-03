import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from database.models import MLModel, SQLMLModel

load_dotenv()

MONGO_URL = os.getenv("DATABASE_URL")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK")

try:
    ALLOW_LOCAL_FALLBACK = bool(int(ALLOW_LOCAL_FALLBACK))
except:
    ALLOW_LOCAL_FALLBACK = False


try:
    client = AsyncIOMotorClient(MONGO_URL)
except:
    client = None


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_model(self, ml_model: MLModel):
        # Verify MongoDB connection, with SQLite as a local fallback (if ALLOW_FALLBACK activated, raise a HTTPException if not the case.)
        try:
            await asyncio.wait_for(client.admin.command("ping"), timeout=2)
            data = ml_model.model_dump()
            result = await client[mongo_db][mongo_collection].insert_one(data)
            new_item_id = await client[mongo_db][mongo_collection].find_one(
                {"_id": str(result.inserted_id)}
            )
        except:
            if ALLOW_LOCAL_FALLBACK is False:
                raise HTTPException(
                    status_code=500, detail="Error when connecting to DB."
                )
            else:
                new_item = SQLMLModel(id=uuid4(), **ml_model.model_dump())
                self.session.add(new_item)
                await self.session.commit()
                new_item_id = new_item.id

        return new_item_id

    async def load_model_from_db(self, model_id: str):
        # If don't connect to Mongo + ALLOW_FALLBACK deactivated, raise a HTTPException
        try:
            await asyncio.wait_for(client.admin.command("ping"), timeout=2)
            result = await client[mongo_db][mongo_collection].find_one(
                {"name": model_id}
            )
        except:
            if ALLOW_LOCAL_FALLBACK is False:
                raise HTTPException(
                    status_code=500, detail="Error when connecting to DB."
                )
            else:
                query = await self.session.execute(
                    select(SQLMLModel).where(SQLMLModel.name == model_id)
                )
                result_orm = query.scalar_one_or_none()
                result = result_orm.model_dump() if result_orm else None

        return result
