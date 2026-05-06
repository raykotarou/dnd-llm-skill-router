from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerSettings(StrictBaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    enable_cors: bool


class LMStudioSettings(StrictBaseModel):
    base_url: str
    api_key: str
    router_model: str
    main_model: str
    request_timeout_seconds: float = Field(default=600.0, gt=0.0)


class RoutingSettings(StrictBaseModel):
    confidence_threshold: int = Field(ge=0, le=100)
    max_ranked_skills: int = Field(ge=1)
    use_manual_commands: bool


class SkillsSettings(StrictBaseModel):
    directory: str
    default_skill: str
    shared_answer_rules: str
    consistency_lens: str


class GenerationSettings(StrictBaseModel):
    router_temperature: float = Field(ge=0.0)
    router_max_tokens: int = Field(ge=1)
    main_temperature: float = Field(ge=0.0)
    main_max_tokens: int = Field(ge=1)
    stream: bool


class LoggingSettings(StrictBaseModel):
    level: str
    log_file: str
    debug_full_payload: bool


class StreamingSettings(StrictBaseModel):
    enabled: bool
    mode: str
    fallback_to_fake_streaming: bool
    lm_studio_timeout_seconds: float = Field(default=600.0, gt=0.0)
    send_done_on_disconnect: bool


class ResponsesReasoningSettings(StrictBaseModel):
    mode: Literal["drop", "think_block", "plain", "pass_through"]
    stream_insertion_strategy: Literal["transform_reasoning_events"]
    preserve_usage: bool
    strip_reasoning_from_completed: bool
    log_presence: bool
    log_raw_reasoning: bool


class ResponsesDiagnosticsSettings(StrictBaseModel):
    enabled: bool
    placement: Literal["start", "end", "both"]
    format: Literal["visible_block", "html_comment"]
    include_source_api: bool
    include_reasoning_mode: bool
    include_streaming_strategy: bool
    include_selected_skill: bool
    include_confidence: bool
    include_manual_skill: bool


class ResponsesAPISettings(StrictBaseModel):
    enabled: bool
    proxy_to_lm_studio_responses: bool
    support_streaming: bool
    support_previous_response_id_passthrough: bool
    store_previous_responses: bool
    unsupported_tools_policy: str = Field(pattern="^(ignore|reject)$")
    reasoning: ResponsesReasoningSettings
    diagnostics: ResponsesDiagnosticsSettings


class AppSettings(StrictBaseModel):
    server: ServerSettings
    lm_studio: LMStudioSettings
    routing: RoutingSettings
    skills: SkillsSettings
    generation: GenerationSettings
    logging: LoggingSettings
    streaming: StreamingSettings
    responses_api: ResponsesAPISettings


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.exists() or path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    project_path = project_root / path
    if project_path.exists():
        return project_path

    return path


def load_settings(config_path: str = "./config.yaml") -> AppSettings:
    path = _resolve_config_path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config: Any = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config at {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    return AppSettings.model_validate(raw_config)
