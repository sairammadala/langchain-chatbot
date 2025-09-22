# chat.py
from typing import Optional, List, Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from handlers.base import BaseHandler

router = APIRouter()

class ChatModel(BaseModel):
    query: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    vector_fetch_k: Optional[int] = 5
    chat_history: List[str] = []
    namespace: Optional[str] = None

def _serialize_source_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs or []:
        # LangChain Document-like
        page_content = getattr(d, "page_content", None)
        metadata = getattr(d, "metadata", None)
        if page_content is not None:
            try:
                out.append({
                    "page_content": page_content,
                    "metadata": dict(metadata or {}),
                })
            except Exception:
                out.append({"page_content": str(page_content), "metadata": {}})
        else:
            # already dict or unknown type
            if isinstance(d, dict):
                out.append({"page_content": d.get("page_content", ""), "metadata": dict(d.get("metadata", {}))})
            else:
                out.append({"page_content": str(d), "metadata": {}})
    return out

@router.post("/chat")
async def chat(chat_model: ChatModel):
    available_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-1106-preview",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
    ]

    if chat_model.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Choose from: {available_models}",
        )

    if not (0.0 <= chat_model.temperature <= 2.0):
        raise HTTPException(
            status_code=400,
            detail="Invalid temperature. Use a value between 0.0 and 2.0.",
        )

    handler = BaseHandler(chat_model=chat_model.model, temperature=chat_model.temperature)
    k = chat_model.vector_fetch_k or 5

    try:
        result = handler.chat(
            query=chat_model.query,
            chat_history=chat_model.chat_history,
            namespace=chat_model.namespace,
            search_kwargs={"k": k},
        )
        # LangChain ConversationalRetrievalChain returns a dict like:
        # {'answer': str, 'source_documents': [Document, ...], ...}
        answer = result.get("answer", "")
        src_docs = _serialize_source_docs(result.get("source_documents", []))

        return {"response": {"answer": answer, "source_documents": src_docs}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
