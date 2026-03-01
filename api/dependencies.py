from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_session
from services.model import ModelService

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_model_service(session: SessionDep):
    return ModelService(session)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
