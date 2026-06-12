from __future__ import annotations

from typing import Any

NESTED_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "decoder_config",
    "model_config",
)


def get_nested_config(config: Any) -> Any:
    """Return text/language sub-config when dimensions are nested."""
    for key in NESTED_CONFIG_KEYS:
        if isinstance(config, dict):
            value = config.get(key)
        else:
            value = getattr(config, key, None)
        if value is not None:
            return value
    return config


def get_config_value(config: Any, names: tuple[str, ...], *, required: bool = True, default: Any = None) -> Any:
    for candidate in (config, get_nested_config(config)):
        for name in names:
            if isinstance(candidate, dict):
                value = candidate.get(name)
            else:
                value = getattr(candidate, name, None)
            if value is not None:
                return value
    if required:
        raise ValueError(f"Could not find config field. Tried: {list(names)}")
    return default


def get_hidden_size(config: Any) -> int:
    return int(get_config_value(config, ("hidden_size", "d_model", "n_embd", "embed_dim")))


def get_num_hidden_layers(config: Any) -> int:
    return int(get_config_value(config, ("num_hidden_layers", "num_layers", "n_layer", "num_decoder_layers")))


def get_vocab_size(config: Any) -> int:
    return int(get_config_value(config, ("vocab_size", "text_vocab_size")))


def get_num_attention_heads(config: Any) -> int | None:
    value = get_config_value(config, ("num_attention_heads", "num_heads", "n_head"), required=False)
    return int(value) if value is not None else None


def get_num_key_value_heads(config: Any) -> int | None:
    value = get_config_value(config, ("num_key_value_heads", "num_kv_heads"), required=False)
    return int(value) if value is not None else None


def get_intermediate_size(config: Any) -> int | None:
    value = get_config_value(config, ("intermediate_size", "ffn_dim", "mlp_dim"), required=False)
    return int(value) if value is not None else None
