from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from gen_ans import format_docs
from embedding_functions import llm
import json
import re

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

model = llm

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
You MUST respond with ONLY a raw JSON object, no explanation, no markdown, no code blocks.
Example: {{"binary_score": "yes"}} or {{"binary_score": "no"}}"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
])

hallucination_grader = hallucination_prompt | model | StrOutputParser()

def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {raw!r}")

def hallucination_check(docs_to_use, generation):
    try:
        raw = hallucination_grader.invoke({
            "documents": format_docs(docs_to_use),
            "generation": generation
        })
        print(f"  [HALLUCINATION] raw='{raw}'")

        parsed = _parse_json(raw)
        binary_score = parsed.get("binary_score", "yes").strip().lower()

        print(f"  [HALLUCINATION] binary_score='{binary_score}'")
        return GradeHallucinations(binary_score=binary_score)

    except Exception as e:
        print(f"  [HALLUCINATION] Parse error: {e} — defaulting to 'yes'")
        return GradeHallucinations(binary_score="yes")