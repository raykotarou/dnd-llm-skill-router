from typing import Any
import json
import time

from langgraph.graph import END, StateGraph

from app.adapters.reasoning_stream_transformer import (
    ResponsesReasoningStreamTransformer,
    build_diagnostics_text,
    transform_non_streaming_response,
)
from app.adapters.sse import format_sse_event, iter_sse_events_from_bytes
from app.config.settings import load_settings
from app.graph.nodes import (
    call_lm_studio,
    detect_manual_skill_command,
    enrich_prompt,
    extract_latest_user_message,
    format_response,
    load_skill_prompt,
    log_routing_decision,
    parse_request,
    rank_skills,
    select_skill_or_clarify,
)
from app.graph.prepared import PreparedGeneration
from app.graph.state import RouterState
from app.lm.lm_studio_client import LMStudioClient


def _initial_state(raw_request: dict[str, Any]) -> RouterState:
    return {
        "raw_request": raw_request,
        "messages": [],
        "latest_user_message": "",
        "manual_skill": None,
        "ranked_skills": [],
        "selected_skill": None,
        "confidence": None,
        "needs_clarification": False,
        "clarification_message": None,
        "skill_prompt": None,
        "answer_rules": None,
        "consistency_lens": None,
        "enriched_messages": [],
        "lm_response": None,
    }


def build_prepare_pipeline(
    confidence_threshold: int,
    max_ranked_skills: int,
    skills_settings,
    logging_settings,
    lm_client,
):
    graph = StateGraph(RouterState)
    graph.add_node("parse_request", parse_request)
    graph.add_node("extract_latest_user_message", extract_latest_user_message)
    graph.add_node("detect_manual_skill_command", detect_manual_skill_command)
    graph.add_node(
        "rank_skills",
        lambda state: rank_skills(state, lm_client, max_ranked_skills),
    )
    graph.add_node(
        "select_skill_or_clarify",
        lambda state: select_skill_or_clarify(state, confidence_threshold),
    )
    graph.add_node(
        "log_routing_decision",
        lambda state: log_routing_decision(state, logging_settings),
    )
    graph.add_node(
        "load_skill_prompt",
        lambda state: load_skill_prompt(state, skills_settings),
    )
    graph.add_node("enrich_prompt", enrich_prompt)
    graph.set_entry_point("parse_request")
    graph.add_edge("parse_request", "extract_latest_user_message")
    graph.add_edge("extract_latest_user_message", "detect_manual_skill_command")
    graph.add_edge("detect_manual_skill_command", "rank_skills")
    graph.add_edge("rank_skills", "select_skill_or_clarify")
    graph.add_edge("select_skill_or_clarify", "log_routing_decision")
    graph.add_edge("log_routing_decision", "load_skill_prompt")
    graph.add_edge("load_skill_prompt", "enrich_prompt")
    graph.add_edge("enrich_prompt", END)
    return graph.compile()


async def prepare_generation(raw_request: dict[str, Any]) -> PreparedGeneration:
    settings = load_settings()
    lm_client = LMStudioClient(settings)
    pipeline = build_prepare_pipeline(
        settings.routing.confidence_threshold,
        settings.routing.max_ranked_skills,
        settings.skills,
        settings.logging,
        lm_client,
    )

    try:
        state = await pipeline.ainvoke(_initial_state(raw_request))
    finally:
        lm_client.close()
        await lm_client.aclose()

    return PreparedGeneration(
        raw_request=raw_request,
        model=raw_request.get("model", "dnd-skill-router"),
        stream=bool(raw_request.get("stream", False)) and settings.streaming.enabled,
        messages=state["messages"],
        enriched_messages=state["enriched_messages"],
        selected_skill=state["selected_skill"],
        confidence=state["confidence"],
        needs_clarification=state["needs_clarification"],
        clarification_message=state["clarification_message"],
        routing_metadata={
            "manual_skill": state["manual_skill"],
            "selected_skill": state["selected_skill"],
            "confidence": state["confidence"],
            "ranked_skills": [
                {"skill": sr.skill, "confidence": sr.confidence}
                for sr in state["ranked_skills"]
            ],
        },
    )


async def complete_generation(prepared: PreparedGeneration) -> dict[str, Any]:
    if prepared.needs_clarification:
        return _build_clarification_response(prepared)

    settings = load_settings()
    lm_client = LMStudioClient(settings)

    try:
        return lm_client.call_main_model(
            messages=prepared.enriched_messages,
            raw_request=prepared.raw_request,
        )
    finally:
        lm_client.close()
        await lm_client.aclose()


async def stream_generation(prepared: PreparedGeneration):
    if prepared.needs_clarification:
        async for chunk in _stream_clarification_response(prepared):
            yield chunk
        return

    settings = load_settings()
    lm_client = LMStudioClient(settings)

    try:
        async for chunk in lm_client.stream_main_model(
            messages=prepared.enriched_messages,
            raw_request=prepared.raw_request,
        ):
            yield chunk
    finally:
        lm_client.close()
        await lm_client.aclose()


async def complete_responses_generation(prepared: PreparedGeneration) -> dict[str, Any]:
    if prepared.needs_clarification:
        return _build_responses_clarification_response(prepared)

    settings = load_settings()
    lm_client = LMStudioClient(settings)

    try:
        response = await lm_client.call_responses_model(
            enriched_messages=prepared.enriched_messages,
            raw_request=prepared.raw_request,
        )
        return transform_non_streaming_response(
            response=response,
            reasoning_mode=settings.responses_api.reasoning.mode,
            diagnostics_config=settings.responses_api.diagnostics.model_dump(),
            routing_metadata=_responses_routing_metadata(prepared),
            strip_reasoning=settings.responses_api.reasoning.strip_reasoning_from_completed,
        )
    finally:
        lm_client.close()
        await lm_client.aclose()


async def stream_responses_generation(prepared: PreparedGeneration):
    if prepared.needs_clarification:
        async for chunk in _stream_responses_clarification_response(prepared):
            yield chunk
        return

    settings = load_settings()
    lm_client = LMStudioClient(settings)
    transformer = ResponsesReasoningStreamTransformer(
        reasoning_mode=settings.responses_api.reasoning.mode,
        stream_insertion_strategy=(
            settings.responses_api.reasoning.stream_insertion_strategy
        ),
        diagnostics_config=settings.responses_api.diagnostics.model_dump(),
        routing_metadata=_responses_routing_metadata(prepared),
        strip_reasoning_from_completed=(
            settings.responses_api.reasoning.strip_reasoning_from_completed
        ),
        log_presence=settings.responses_api.reasoning.log_presence,
        log_raw_reasoning=settings.responses_api.reasoning.log_raw_reasoning,
    )

    try:
        upstream_events = iter_sse_events_from_bytes(
            lm_client.stream_responses_model(
                enriched_messages=prepared.enriched_messages,
                raw_request=prepared.raw_request,
            )
        )
        async for upstream_event in upstream_events:
            for downstream_event in transformer.transform(upstream_event):
                if downstream_event.data is not None:
                    yield format_sse_event(
                        downstream_event.event,
                        downstream_event.data,
                    )
    except Exception as exc:
        yield _responses_error_event(f"LM Studio stream interrupted: {exc}")
    finally:
        lm_client.close()
        await lm_client.aclose()


def _build_clarification_response(prepared: PreparedGeneration) -> dict[str, Any]:
    return {
        "id": "chatcmpl-local-clarification",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": prepared.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": prepared.clarification_message,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def _stream_clarification_response(prepared: PreparedGeneration):
    base_chunk = {
        "id": "chatcmpl-local-clarification",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": prepared.model,
    }

    # Role chunk
    role_chunk = {
        **base_chunk,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    # Content chunk
    if prepared.clarification_message:
        content_chunk = {
            **base_chunk,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": prepared.clarification_message},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

    # Final chunk
    final_chunk = {
        **base_chunk,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    # Done
    yield "data: [DONE]\n\n"


def _build_responses_clarification_response(prepared: PreparedGeneration) -> dict[str, Any]:
    message = _responses_clarification_text(prepared)
    return {
        "id": "resp_local_clarification",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": prepared.model,
        "output": [
            {
                "id": "msg_local_clarification",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": message,
                    }
                ],
            }
        ],
        "usage": None,
    }


async def _stream_responses_clarification_response(prepared: PreparedGeneration):
    response_id = "resp_local_clarification"
    message = _responses_clarification_text(prepared)

    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": prepared.model,
        },
    }
    yield _responses_event("response.created", created_event)

    if message:
        delta_event = {
            "type": "response.output_text.delta",
            "item_id": "msg_local_clarification",
            "output_index": 0,
            "content_index": 0,
            "delta": message,
        }
        yield _responses_event("response.output_text.delta", delta_event)

    completed_event = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "model": prepared.model,
        },
    }
    yield _responses_event("response.completed", completed_event)


def _responses_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _responses_error_event(message: str) -> str:
    return _responses_event(
        "error",
        {
            "type": "error",
            "message": message,
        },
    )


def _responses_routing_metadata(prepared: PreparedGeneration) -> dict[str, Any]:
    return {
        **prepared.routing_metadata,
        "selected_skill": prepared.selected_skill,
        "confidence": prepared.confidence,
    }


def _responses_clarification_text(prepared: PreparedGeneration) -> str:
    message = prepared.clarification_message or ""
    settings = load_settings()
    diagnostics = settings.responses_api.diagnostics
    if not diagnostics.enabled:
        return message

    diagnostics_text = build_diagnostics_text(
        source_api="responses",
        reasoning_mode=settings.responses_api.reasoning.mode,
        stream_insertion_strategy=(
            settings.responses_api.reasoning.stream_insertion_strategy
        ),
        selected_skill=prepared.selected_skill,
        confidence=prepared.confidence,
        manual_skill=prepared.routing_metadata.get("manual_skill"),
        format=diagnostics.format,
    )
    if diagnostics.placement == "end":
        return f"{message}\n\n{diagnostics_text}"
    if diagnostics.placement == "both":
        return f"{diagnostics_text}\n\n{message}\n\n{diagnostics_text}"
    return f"{diagnostics_text}\n\n{message}"


# Backward compatibility
build_pipeline = build_prepare_pipeline
