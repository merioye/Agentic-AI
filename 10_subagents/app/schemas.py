from pydantic import BaseModel, Field
import uuid


def message_content_to_text(content: object) -> str:
    """Convert LangChain/provider message content into response text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


class SupportRequest(BaseModel):
    customer_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Authenticated customer identifier",
    )
    message: str = Field(
        ..., min_length=1, max_length=4000,
        description="The customer's support message",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Groups messages into a conversation; client can pass an "
                    "existing one to continue a thread, or omit it to start fresh",
    )


class SupportResponse(BaseModel):
    session_id: str
    reply: str
    subagents_used: list[str] = Field(
        default_factory=list,
        description="Which specialist subagents were consulted, for transparency/debugging",
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
