"""CrewAI wiring for the five required agent roles (spec §15).

Two CrewAI constructs live here:

- `run_request_flow` -- the LIVE request path. orchestrator.process_request executes every
  customer request as a CrewAI Flow (`NexusRequestFlow`): five Flow steps, one per Agent role
  (Classifier -> Retriever -> Drafting Agent -> Validation Agent -> Escalation Agent), chained
  avec @start/@listen. Each step runs that role's stage of the pipeline; the Escalation step
  still defers to the deterministic MCP run_escalation_check, which remains the sole
  decision-maker (Invariant 5).
- `build_agents` / `build_crew` -- the same five roles as LLM-driven CrewAI Agents/Tasks bound to
  tools, for an autonomous `Crew.kickoff()` when an API key is configured (used by tests and for
  exploration; customer requests use the Flow because its control flow is deterministic).

Each CrewAI Agent is bound to a single tool that delegates to the deterministic/LLM-backed
function in agents/classifier.py, agents/retriever.py, agents/drafting.py, agents/validation.py,
agents/escalation.py. This keeps CrewAI responsible for orchestration/role framing while the
actual decisions -- especially escalation -- stay in deterministic Python code, per the
spec's final architecture principle (§58): LLMs propose, deterministic code enforces.
"""

import os

# CrewAI ships with OpenTelemetry-based usage telemetry that posts to an external endpoint at
# import/run time. A financial-services concierge must not phone home, so opt out before the
# package is imported (setdefault keeps an explicit operator choice).
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# No execution-trace upload and no interactive "view your traces?" prompt in the request path.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from agents import anthropic_client
from agents.classifier import classify
from agents.drafting import draft_response
from agents.escalation import check_escalation
from agents.retriever import retrieve_grounded_context
from agents.validation import validate_draft
from config import model_config


class _OfflineStubLLM(BaseLLM):
    """Stand-in LLM used when ANTHROPIC_API_KEY is unset.

    CrewAI validates the LLM when an Agent is constructed, so without this the five required
    agents could not even be instantiated offline. The deterministic pipeline in
    orchestrator.py is what actually runs in that case.
    """

    def __init__(self, model: str = "offline-stub"):
        super().__init__(model=model)

    def call(
        self, messages, tools=None, callbacks=None, available_functions=None, **kwargs
    ):
        return (
            "ANTHROPIC_API_KEY is not configured, so this CrewAI agent cannot perform LLM "
            "reasoning. Use the deterministic Nexus pipeline (orchestrator.process_request)."
        )


def _llm():
    if anthropic_client.is_configured():
        return f"anthropic/{model_config.CLASSIFIER_MODEL}"
    return _OfflineStubLLM()


@tool("classify_request")
def classify_tool(sanitized_text: str, customer_id: str) -> str:
    """Classify a sanitized customer message into a structured intake JSON string."""
    intake = classify(sanitized_text, customer_id)
    return intake.model_dump_json()


@tool("retrieve_grounded_context")
def retrieve_tool(intake_json: str) -> str:
    """Retrieve hybrid grounded context for a validated intake JSON string."""
    from schemas.intake import IntakeRequest

    intake = IntakeRequest.model_validate_json(intake_json)
    context = retrieve_grounded_context(intake)
    return context.model_dump_json()


@tool("draft_customer_response")
def draft_tool(intake_json: str, context_json: str) -> str:
    """Draft a grounded customer response from intake + grounded context JSON strings."""
    from schemas.intake import IntakeRequest
    from schemas.retrieval import GroundedContext

    intake = IntakeRequest.model_validate_json(intake_json)
    context = GroundedContext.model_validate_json(context_json)
    return draft_response(intake, context)


@tool("validate_draft_response")
def validate_tool(intake_json: str, draft: str, context_json: str) -> str:
    """Validate a draft response against grounded context, returning a ValidationResult JSON string."""
    from schemas.intake import IntakeRequest
    from schemas.retrieval import GroundedContext

    intake = IntakeRequest.model_validate_json(intake_json)
    context = GroundedContext.model_validate_json(context_json)
    result = validate_draft(intake, draft, context)
    return result.model_dump_json()


@tool("check_escalation")
def escalation_tool(intake_json: str, validation_json: str) -> str:
    """Run the deterministic escalation check for an intake + validation result."""
    import json

    from schemas.intake import IntakeRequest
    from schemas.validation import ValidationResult

    intake = IntakeRequest.model_validate_json(intake_json)
    validation = ValidationResult(**json.loads(validation_json))
    decision = check_escalation(intake, validation)
    return json.dumps(decision)


def build_agents() -> dict:
    llm = _llm()
    return {
        "classifier": Agent(
            role="Classifier",
            goal="Convert sanitized customer requests into validated structured intake.",
            backstory="A precise intake specialist who never guesses an invalid field.",
            tools=[classify_tool],
            llm=llm,
            allow_delegation=False,
        ),
        "retriever": Agent(
            role="Retriever",
            goal="Retrieve and reconcile structured MCP data and unstructured knowledge-base data.",
            backstory="A careful researcher who never drafts the final answer.",
            tools=[retrieve_tool],
            llm=llm,
            allow_delegation=False,
        ),
        "drafter": Agent(
            role="Drafting Agent",
            goal="Draft a customer response using only grounded context.",
            backstory="A grounded writer who never invents facts.",
            tools=[draft_tool],
            llm=llm,
            allow_delegation=False,
        ),
        "validator": Agent(
            role="Validation Agent",
            goal="Verify the draft is factually grounded, compliant, and scored with confidence.",
            backstory="A skeptical reviewer who flags unsupported claims and conflicts.",
            tools=[validate_tool],
            llm=llm,
            allow_delegation=False,
        ),
        "escalation": Agent(
            role="Escalation Agent",
            goal="Apply the deterministic escalation decision without overriding it.",
            backstory="A rules-bound compliance officer who always defers to run_escalation_check.",
            tools=[escalation_tool],
            llm=llm,
            allow_delegation=False,
        ),
    }


def build_crew(agents: dict, sanitized_text: str, customer_id: str) -> Crew:
    classify_task = Task(
        description=f"Classify this sanitized customer message for customer {customer_id}: {sanitized_text}",
        expected_output="A structured intake JSON string.",
        agent=agents["classifier"],
    )
    retrieve_task = Task(
        description="Retrieve grounded context for the classified intake produced in the previous task.",
        expected_output="A GroundedContext JSON string.",
        agent=agents["retriever"],
        context=[classify_task],
    )
    draft_task = Task(
        description="Draft a customer response using only the grounded context from the previous task. If out of scope, then deny gracefully.",
        expected_output="A plain-text customer response.",
        agent=agents["drafter"],
        context=[classify_task, retrieve_task],
    )
    validate_task = Task(
        description="Validate the draft against the grounded context and score confidence.",
        expected_output="A ValidationResult JSON string.",
        agent=agents["validator"],
        context=[classify_task, retrieve_task, draft_task],
    )
    escalate_task = Task(
        description="Run the deterministic escalation check using the intake and validation result.",
        expected_output="An escalation decision JSON string.",
        agent=agents["escalation"],
        context=[classify_task, validate_task],
    )

    return Crew(
        agents=list(agents.values()),
        tasks=[classify_task, retrieve_task, draft_task, validate_task, escalate_task],
        process=Process.sequential,
    )


# =============================================================================================
# Live request path: CrewAI Flow
# =============================================================================================
# Flow step -> the CrewAI Agent role (Agent(role=...) above) that owns it.
FLOW_STEPS: tuple[str, ...] = (
    "classifier",
    "retriever",
    "drafter",
    "validator",
    "escalation",
)
AGENT_ROLES: dict[str, str] = {
    "classifier": "Classifier",
    "retriever": "Retriever",
    "drafter": "Drafting Agent",
    "validator": "Validation Agent",
    "escalation": "Escalation Agent",
}


class CrewFlowUnavailableError(RuntimeError):
    """CrewAI's Flow runtime could not run (import/initialisation failure) -- not a stage error."""


class _FlowRun:
    def __init__(self, stages: dict):
        self.stages = stages
        self.completed: list[str] = []
        self.error: BaseException | None = None

    def execute(self, step: str):
        if (
            self.error is not None
        ):  # an earlier stage failed: every later stage is skipped
            return None
        stage = self.stages.get(step)
        if stage is None:
            return None
        try:
            stage()
        except (
            BaseException
        ) as exc:  # stored and re-raised in the caller's thread, unchanged
            self.error = exc
            return None
        self.completed.append(step)
        return AGENT_ROLES[step]


_FLOW_CLASS = None


def _flow_class():
    """Build the Flow class once (deferred so importing this module never requires crewai.flow)."""
    global _FLOW_CLASS
    if _FLOW_CLASS is not None:
        return _FLOW_CLASS
    from crewai.flow.flow import Flow, listen, start

    class NexusRequestFlow(Flow):
        """One customer request = one Flow run; each step is owned by one CrewAI Agent role."""

        def _run(self, step):
            return self.__dict__["_nexus_run"].execute(step)

        @start()
        def classifier_step(self):
            return self._run("classifier")

        @listen(classifier_step)
        def retriever_step(self):
            return self._run("retriever")

        @listen(retriever_step)
        def drafter_step(self):
            return self._run("drafter")

        @listen(drafter_step)
        def validator_step(self):
            return self._run("validator")

        @listen(validator_step)
        def escalation_step(self):
            return self._run("escalation")

    _FLOW_CLASS = NexusRequestFlow
    return _FLOW_CLASS


def run_request_flow(stages: dict) -> list[str]:
    """Execute the request's five stages as a CrewAI Flow; returns the roles that completed.

    `stages` maps each FLOW_STEPS name to a zero-argument callable. A stage exception is re-raised
    exactly as raised (typed Nexus errors keep their meaning). If the Flow runtime itself fails,
    CrewFlowUnavailableError carries the steps that did complete so the caller can finish the
    remaining ones directly without repeating side effects.
    """
    run = _FlowRun(stages)
    try:
        flow = _flow_class()()
        flow.__dict__["_nexus_run"] = run
        flow.kickoff()
    except BaseException as exc:
        if run.error is not None:
            raise run.error
        error = CrewFlowUnavailableError(
            f"CrewAI Flow runtime failed: {type(exc).__name__}: {exc}"
        )
        error.completed = list(run.completed)
        raise error from exc
    if run.error is not None:
        raise run.error
    return list(run.completed)
