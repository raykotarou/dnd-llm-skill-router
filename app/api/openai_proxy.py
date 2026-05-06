from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.adapters.responses_adapter import responses_to_canonical
from app.config.settings import load_settings
from app.graph.pipeline import (
    complete_generation,
    complete_responses_generation,
    prepare_generation,
    stream_generation,
    stream_responses_generation,
)


router = APIRouter(prefix="/v1", tags=["openai-proxy"])


@router.get("/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": "dnd-skill-router",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions")
async def create_chat_completion(request: Request):
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid OpenAI chat completion payload: request body must be an object.",
        )

    try:
        prepared = await prepare_generation(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not prepared.stream:
        return await complete_generation(prepared)

    return StreamingResponse(
        stream_generation(prepared),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/responses")
async def create_response(request: Request):
    settings = load_settings()
    if not settings.responses_api.enabled:
        raise HTTPException(status_code=404, detail="Responses API endpoint is disabled.")

    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid Responses payload: request body must be an object.",
        )

    try:
        canonical = responses_to_canonical(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if (
        canonical.unsupported_fields
        and settings.responses_api.unsupported_tools_policy == "reject"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Responses payload contains unsupported fields for MVP: "
                f"{canonical.unsupported_fields}"
            ),
        )

    if canonical.unsupported_fields:
        logger.warning(
            "Responses payload contains unsupported fields for MVP: {}; ignoring them",
            canonical.unsupported_fields,
        )

    stream_requested = canonical.stream
    if not settings.responses_api.support_streaming:
        canonical = canonical.model_copy(update={"stream": False})

    try:
        prepared = await prepare_generation(canonical.to_chat_payload())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.bind(event="responses_generation").info(
        "{}",
        {
            "endpoint": "/v1/responses",
            "source_api": "responses",
            "stream_requested": stream_requested,
            "selected_skill": prepared.selected_skill,
            "confidence": prepared.confidence,
            "needs_clarification": prepared.needs_clarification,
            "previous_response_id_present": canonical.previous_response_id is not None,
            "unsupported_fields_present": canonical.unsupported_fields,
        },
    )

    if not prepared.stream:
        try:
            return await complete_responses_generation(prepared)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        stream_responses_generation(prepared),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
