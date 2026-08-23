"""Pydantic models defining the API's request/response contracts."""
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    conversation_id: str = Field(
        ..., description="Stable ID for this conversation thread. Reuse it across turns."
    )
    message: str = Field(..., min_length=1, max_length=4000)

class ChatResponse(BaseModel):
    message: str
    escalate_to_human: bool
    requires_approval: bool = Field(
        default=False, description="True if a tool call is pending human approval."
    )
    pending_action: str | None = Field(
        default=None, description="Description of the action awaiting approval, if any."
    )

class ResumeRequest(BaseModel):
    conversation_id: str
    decision: str = Field(..., description="'approve' or 'reject'")

class HealthResponse(BaseModel):
    status: str
    model: str