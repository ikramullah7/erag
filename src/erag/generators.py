"""Explicit answer-provider selection; never silently fall back or send local data out."""


def create_generator(cfg):
    if cfg.provider == "codex":
        from .codex import CodexGenerator

        return CodexGenerator(cfg)
    if cfg.provider != "ollama":
        raise ValueError("unknown answer provider")
    from .ollama import OllamaGenerator

    return OllamaGenerator(cfg)
