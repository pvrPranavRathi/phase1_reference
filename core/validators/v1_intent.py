from core.ir.ir1 import ir1_intent

class ir1_validation_error(Exception):
    pass

def validate_ir1(ir1: ir1_intent):
    # Rule 1: teaching_objective must be non-trivial
    if len(ir1.teaching_objective.split()) < 15:
        raise ir1_validation_error("teaching_objective is too short or vague.")
    
    # Rule 2: required_priors must not be empty
    if not ir1.required_priors:
        raise ir1_validation_error("required_priors cannot be empty.")
    
    # Rule 3: priors must be concept-level, not section-level
    forbidden_terms = ["section", "paper", "method", "architecture"]
    for prior in ir1.required_priors:
        for term in forbidden_terms:
            if term in prior.lower():
                raise ir1_validation_error(f"Invalid prior detected: '{prior}'")