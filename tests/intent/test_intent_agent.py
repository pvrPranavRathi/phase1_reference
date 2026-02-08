import pytest

from core.agents.intent import (
    run_section_intent_agent,
    section_intent_agent_error
)
from core.ir.ir1 import ir1_intent

def test_section_intent_agent_happy_path():
    section_text = """We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
    queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
    3
    Scaled Dot-Product Attention Multi-Head Attention
    Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several
    attention layers running in parallel.
    query with all keys, divide each by √
    dk, and apply a softmax function to obtain the weights on the
    values.
    In practice, we compute the attention function on a set of queries simultaneously, packed together
    into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
    the matrix of outputs as:
    Attention(Q, K, V ) = softmax(QKT
    √
    dk
    )V (1)
    The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
    of √
    1
    dk
    . Additive attention computes the compatibility function using a feed-forward network with
    a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
    much faster and more space-efficient in practice, since it can be implemented using highly optimized
    matrix multiplication code.
    While for small values of dk the two mechanisms perform similarly, additive attention outperforms
    dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
    dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
    extremely small gradients 4
    . To counteract this effect, we scale the dot products by √
    1
    dk
    .
    """

    def mock_llm_call(prompt: str) -> str:
        return """
        {
          "section_role": "mechanism_definition",
          "teaching_objective": "explain how attention weights are computed from queries and keys to combine information",
          "required_priors": [
            "vector representations",
            "dot product similarity",
            "probability normalization"
          ]
        }
        """

    ir1 = run_section_intent_agent(section_text, mock_llm_call)

    assert isinstance(ir1, ir1_intent)
    assert ir1.section_role == "mechanism_definition"
    assert "attention weights" in ir1.teaching_objective
    assert len(ir1.required_priors) >= 2

def test_section_intent_agent_invalid_role():
    def mock_llm_call(prompt: str) -> str:
        return """
        {
          "section_role": "math_derivation",
          "teaching_objective": "explain attention",
          "required_priors": ["vectors"]
        }
        """

    with pytest.raises(section_intent_agent_error):
        run_section_intent_agent("text", mock_llm_call)

def test_section_intent_agent_missing_field():
    def mock_llm_call(prompt: str) -> str:
        return """
        {
          "section_role": "mechanism_definition",
          "teaching_objective": "explain attention"
        }
        """

    with pytest.raises(section_intent_agent_error):
        run_section_intent_agent("text", mock_llm_call)

def test_section_intent_agent_non_json_output():
    def mock_llm_call(prompt: str) -> str:
        return "Here is the explanation of the section..."

    with pytest.raises(section_intent_agent_error):
        run_section_intent_agent("text", mock_llm_call)


from core.validators.v1_intent import validate_ir1, ir1_validation_error

def test_ir1_validator_rejects_vague_objective():
    ir1 = ir1_intent(
        section_role="mechanism_definition",
        teaching_objective="explain attention",
        required_priors=["vectors"]
    )

    with pytest.raises(ir1_validation_error):
        validate_ir1(ir1)
