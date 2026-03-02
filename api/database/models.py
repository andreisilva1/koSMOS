from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from bson import ObjectId
from pydantic import BaseModel
from sqlmodel import Field, SQLModel


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError()
        return ObjectId(v)


class MLModel(BaseModel):
    name: str
    model: bytes
    preprocessor: bytes
    dict_types: str
    target: Optional[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SQLMLModel(SQLModel, table=True):
    id: UUID = Field(primary_key=True)
    name: str
    model: bytes
    preprocessor: bytes
    dict_types: str
    target: str = Field(nullable=True)
    created_at: datetime
