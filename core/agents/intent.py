from core.ir.ir1 import ir1_intent
from pydantic import ValidationError



class section_intent_agent_error(Exception):
    pass



INTENT_PROMPT_TEMPLATE = """
You are a Section Intent Analysis Agent.

Your task is to analyze a single research paper section and determine:
1. The logical role this section plays in the paper.
2. The core teaching objective this section is meant to install in the reader.
3. The minimal set of prerequisite concepts required to understand this section.

Constraints (STRICT):
- Do NOT mention equations, symbols, or variable names.
- Do NOT mention implementation details.
- Do NOT mention diagrams or visuals.
- Do NOT summarize the section.
- Do NOT anticipate later sections of the paper.

Allowed section_role values:
- mechanism_definition
- mechanism_composition
- conceptual_setup
- architectural_overview

Output format:
Return a JSON object exactly matching this schema:

{{
  "section_role": "<one of the allowed values>",
  "teaching_objective": "<single concise sentence>",
  "required_priors": ["<prior 1>", "<prior 2>", "..."]
}}

Section text:
\"\"\"
{section_text}
\"\"\"
"""



def run_section_intent_agent(section_text: str, llm_call) -> ir1_intent:
    """
    Runs the SectionIntentAgent.

    llm_call: callable(prompt: str) -> str
      Must return a raw JSON string matching ir1_intent schema.
    """

    prompt = build_internal_prompt(section_text)

    try:
        raw_output = llm_call(prompt)
        return ir1_intent.model_validate_json(raw_output)
    except ValidationError as e:
        raise section_intent_agent_error(f"Failed to parse IR-1 from LLM output: {e}")
    except Exception as e:
        raise section_intent_agent_error(f"SectionIntentAgent execution failed: {e}")



def build_internal_prompt(section_text: str) -> str:
    return INTENT_PROMPT_TEMPLATE.format(section_text=section_text)