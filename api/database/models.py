from typing import Optional

from bson import ObjectId
from pydantic import BaseModel

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
    