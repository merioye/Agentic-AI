from pydantic import BaseModel, Field


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)


class ReportSectionOut(BaseModel):
    title: str
    content: str


class TodoOut(BaseModel):
    content: str
    status: str


class TodoChatResponse(BaseModel):
    message: str
    escalate_to_human: bool = False
    report_sections: list[ReportSectionOut] = Field(default_factory=list)
    todos: list[TodoOut] = Field(default_factory=list)
    todos_completed: int = 0
    todos_total: int = 0


class PlanStepOut(BaseModel):
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None


class PlanResponse(BaseModel):
    goal: str
    plan: list[PlanStepOut]


class RunPlanRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    plan: list[PlanStepOut] | None = Field(
        default=None, description="Optional: an already-approved (possibly human-edited) plan from POST /plan-execute/plan. Omit to plan fresh."
    )


class RunPlanResponse(BaseModel):
    goal: str
    final_message: str | None
    plan: list[PlanStepOut]
    report_sections: list[ReportSectionOut]
    replan_cycles: int


class HealthResponse(BaseModel):
    status: str
    model: str