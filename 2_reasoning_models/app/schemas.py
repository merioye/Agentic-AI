from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str = Field(
        default="default", description="For logging/tracing only - this project has no multi-turn memory."
    )
    force_effort: str | None = Field(
        default=None, description="Optional: override the router and force a specific effort level (for testing/comparison)."
    )


class ChatResponse(BaseModel):
    message: str
    escalate_to_human: bool = False
    effort_used: str
    reasoning_summary: str | None = None
    thinking_tokens: int | None = None

class HealthResponse(BaseModel):
    status: str
    model: str