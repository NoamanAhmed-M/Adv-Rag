from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from config import NVIDIA_API_KEY, LLM_MODEL
import os

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

def get_embedding_function():
    return NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=NVIDIA_API_KEY,
        truncate="END",
    )

llm = ChatNVIDIA(
    model=LLM_MODEL,
    api_key=NVIDIA_API_KEY,
    temperature=0,
)

print(f"[LLM] Using model: {LLM_MODEL}")