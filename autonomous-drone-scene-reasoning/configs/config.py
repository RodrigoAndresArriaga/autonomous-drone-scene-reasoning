# Config loader: YAML defaults + CONFIG_PROFILE merge + env overrides.
# Single source of truth for Cosmos model and scene agent runtime settings.

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent

try:
    import yaml
except ImportError:
    yaml = None


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Modifies base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _apply_env_overrides(data: dict) -> None:
    """Apply environment variable overrides to data in place."""
    # TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE -> cosmos.offline
    if os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1":
        data.setdefault("cosmos", {})["offline"] = True

    # COSMOS_TIMING
    if os.environ.get("COSMOS_TIMING"):
        data.setdefault("cosmos", {})["timing"] = True

    # COSMOS_JSON_REPAIR
    if os.environ.get("COSMOS_JSON_REPAIR") == "1":
        data.setdefault("cosmos", {})["json_repair"] = True

    # COSMOS_NO_RESIZE
    if os.environ.get("COSMOS_NO_RESIZE") == "1":
        data.setdefault("cosmos", {}).setdefault("image", {})["resize"] = False

    # COSMOS_MAX_NEW_TOKENS_EXTRACT_IMAGE
    if v := os.environ.get("COSMOS_MAX_NEW_TOKENS_EXTRACT_IMAGE"):
        data.setdefault("cosmos", {}).setdefault("image", {})["max_new_tokens"] = int(v)

    # COSMOS_MAX_NEW_TOKENS_EXTRACT_VIDEO
    if v := os.environ.get("COSMOS_MAX_NEW_TOKENS_EXTRACT_VIDEO"):
        data.setdefault("cosmos", {}).setdefault("video", {})["max_new_tokens"] = int(v)

    # COSMOS_MAX_NEW_TOKENS_NORMALIZE
    if v := os.environ.get("COSMOS_MAX_NEW_TOKENS_NORMALIZE"):
        data.setdefault("cosmos", {})["max_new_tokens_normalize"] = int(v)

    # COSMOS_MAX_NEW_TOKENS_EXPLANATION
    if v := os.environ.get("COSMOS_MAX_NEW_TOKENS_EXPLANATION"):
        data.setdefault("cosmos", {})["max_new_tokens_explanation"] = int(v)

    # COSMOS_MAX_NEW_TOKENS_REPAIR
    if v := os.environ.get("COSMOS_MAX_NEW_TOKENS_REPAIR"):
        data.setdefault("cosmos", {})["max_new_tokens_repair"] = int(v)

    # COSMOS_EVAL_LOG_EVERY
    if v := os.environ.get("COSMOS_EVAL_LOG_EVERY"):
        data.setdefault("agent", {})["eval_log_every"] = int(v)

    # COSMOS_CLIP_SECONDS
    if v := os.environ.get("COSMOS_CLIP_SECONDS"):
        data.setdefault("agent", {})["clip_seconds"] = float(v)

    # COSMOS_STEP_SECONDS
    if v := os.environ.get("COSMOS_STEP_SECONDS"):
        data.setdefault("agent", {})["step_seconds"] = float(v)


# --- Dataclasses ---


@dataclass
class CosmosImageConfig:
    resize: bool
    resize_hw: tuple[int, int]
    max_new_tokens: int


@dataclass
class CosmosVideoConfig:
    fps: int
    max_new_tokens: int


@dataclass
class CosmosGenerationConfig:
    do_sample: bool
    use_cache: bool


@dataclass
class CosmosConfig:
    model_name: str
    offline: bool
    timing: bool
    json_repair: bool
    image: CosmosImageConfig
    video: CosmosVideoConfig
    generation: CosmosGenerationConfig
    attention_preference: list[str]
    max_new_tokens_normalize: int
    max_new_tokens_explanation: int
    max_new_tokens_repair: int


@dataclass
class AgentMemoryConfig:
    max_memory: int


@dataclass
class AgentExplainConfig:
    cache_on_state_signature: bool


@dataclass
class AgentConfig:
    fps_default: int
    eval_log_every: int
    clip_seconds: float
    step_seconds: float
    memory: AgentMemoryConfig
    explain: AgentExplainConfig


@dataclass
class AppConfig:
    cosmos: CosmosConfig
    agent: AgentConfig


def _dict_to_config(data: dict) -> AppConfig:
    c = data.get("cosmos", {})
    ci = c.get("image", {})
    cv = c.get("video", {})
    cg = c.get("generation", {})
    a = data.get("agent", {})
    am = a.get("memory", {})
    ae = a.get("explain", {})

    return AppConfig(
        cosmos=CosmosConfig(
            model_name=c.get("model_name", "nvidia/Cosmos-Reason2-2B"),
            offline=c.get("offline", False),
            timing=c.get("timing", False),
            json_repair=c.get("json_repair", False),
            image=CosmosImageConfig(
                resize=ci.get("resize", True),
                resize_hw=tuple(ci.get("resize_hw", [448, 448])),
                max_new_tokens=int(ci.get("max_new_tokens", 200)),
            ),
            video=CosmosVideoConfig(
                fps=int(cv.get("fps", 4)),
                max_new_tokens=int(cv.get("max_new_tokens", 512)),
            ),
            generation=CosmosGenerationConfig(
                do_sample=cg.get("do_sample", False),
                use_cache=cg.get("use_cache", True),
            ),
            attention_preference=c.get("attention_preference", ["flash_attention_2", "sdpa"]),
            max_new_tokens_normalize=int(c.get("max_new_tokens_normalize", 200)),
            max_new_tokens_explanation=int(c.get("max_new_tokens_explanation", 200)),
            max_new_tokens_repair=int(c.get("max_new_tokens_repair", 200)),
        ),
        agent=AgentConfig(
            fps_default=int(a.get("fps_default", 4)),
            eval_log_every=int(a.get("eval_log_every", 10)),
            clip_seconds=float(a.get("clip_seconds", 2.0)),
            step_seconds=float(a.get("step_seconds", 2.0)),
            memory=AgentMemoryConfig(max_memory=int(am.get("max_memory", 5))),
            explain=AgentExplainConfig(cache_on_state_signature=ae.get("cache_on_state_signature", True)),
        ),
    )


_cached_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Load and cache config. YAML defaults + CONFIG_PROFILE merge + env overrides."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    default_path = _CONFIG_DIR / "default.yaml"
    data = _load_yaml(default_path)

    profile = os.environ.get("CONFIG_PROFILE")
    if profile:
        profile_path = _CONFIG_DIR / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"CONFIG_PROFILE={profile} but {profile_path} not found")
        profile_data = _load_yaml(profile_path)
        _deep_merge(data, profile_data)

    _apply_env_overrides(data)
    _cached_config = _dict_to_config(data)
    return _cached_config
