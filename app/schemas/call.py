from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class CallCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)

    call_date: datetime
    transcript: str = Field(..., min_length=50)


class CallResponse(BaseModel):
    id: UUID
    title: str
    description: str | None
    call_date: datetime
    status: str
    created_at: datetime
