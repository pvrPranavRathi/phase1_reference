from pydantic import BaseModel
from typing import Literal, List

class ir1_intent(BaseModel):
    # section_role is closed vocabulary → prevents drift
    section_role: Literal[
        "mechanism_definition",
        "mechanism_composition",
        "conceptual_setup",
        "architectural_overview"
    ]
    # teaching_objective is single-sentence → forces clarity
    teaching_objective: str
    # required_priors is explicit → prevents hidden assumptions
    required_priors: List[str]