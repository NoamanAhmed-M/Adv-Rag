from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from embedding_functions import llm
import json
import re

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

model = llm

system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
You MUST respond with ONLY a raw JSON object, no explanation, no markdown, no code blocks.
Example: {{"binary_score": "yes"}} or {{"binary_score": "no"}}"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | model | StrOutputParser()

def _parse_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output even if wrapped in markdown."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Last resort: find first {...} block
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {raw!r}")

def relevancy_check(question, docs):
    docs_to_use = []
    for i, (doc, score) in enumerate(docs):
        try:
            raw = retrieval_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            print(f"  [GRADER doc{i+1}] raw='{raw}'")

            parsed = _parse_json(raw)
            binary_score = parsed.get("binary_score", "yes").strip().lower()

            print(f"  [GRADER doc{i+1}] binary_score='{binary_score}'")
            if binary_score == "yes":
                print(f"  [GRADER doc{i+1}] ✅ PASSED")
                docs_to_use.append(doc)
            else:
                print(f"  [GRADER doc{i+1}] ❌ REJECTED")

        except Exception as e:
            print(f"  [GRADER doc{i+1}] ❌ Parse error: {e} — defaulting to 'yes'")
            docs_to_use.append(doc)

    return docs_to_use