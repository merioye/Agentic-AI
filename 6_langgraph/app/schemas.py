from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LineItemIn(BaseModel):
    category: str = Field(..., min_length=1, max_length=40)
    amount: float = Field(..., gt=0, le=100_000)
    description: str = Field(..., min_length=1, max_length=300)
    has_receipt: bool = False

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.strip().lower()


class ExpenseSubmitRequest(BaseModel):
    employee: str = Field(..., min_length=1, max_length=1000)
    line_items: list[LineItemIn] = Field(..., min_length=1, max_length=100)


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    decision: Literal["approved", "rejected"]
    note: str = Field(default="", max_length=500)