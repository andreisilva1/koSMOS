import os
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select
from database.session import get_session
from services.model import ModelService
from checks.database import check_mongo_connection
from database.models import MLModel, SQLMLModel

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_model_service(session: SessionDep):
    return ModelService(session)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
