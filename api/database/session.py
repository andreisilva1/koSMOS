import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from database.models import MLModel

load_dotenv()

MONGO_URL = os.getenv("DATABASE_URL")
mongo_db = os.getenv("MONGO_DB")
mongo_collection = os.getenv("MONGO_COLLECTION")

client = AsyncIOMotorClient(MONGO_URL)
db = client[mongo_db]
collection = db[mongo_collection]

async def save_model(ml_model: MLModel):
    data = ml_model.model_dump()
    result = await collection.insert_one(data)
    new = await collection.find_one({"_id": str(result.inserted_id)})
    return new

async def load_model_from_db(model_id: str):
    result = await collection.find_one({"name": model_id})
    return result
    