from src.memory_rewrite import MemoryCommandRewriter


def _build_rewriter(llm_rewrite=None):
    return MemoryCommandRewriter(
        normalize_alias_key=lambda value: str(value or "").strip().lower(),
        llm_rewrite=llm_rewrite,
    )


def test_memory_rewriter_reverses_device_code_mapping_direction() -> None:
    rewriter = _build_rewriter()
    command = rewriter.rewrite_create_command("请记住以后a1_b9代表一号设备")
    assert command is not None
    assert command["alias_text"] == "一号设备"
    assert command["target_text"] == "a1_b9"
    assert command["reversed_mapping"] is True
    assert command["rewrite_source"] == "rule"


def test_memory_rewriter_parses_natural_phrase_rule_first() -> None:
    rewriter = _build_rewriter()
    command = rewriter.rewrite_create_command("帮我记一下研发动力表其实就是七层动力表")
    assert command is not None
    assert command["alias_text"] == "研发动力表"
    assert command["target_text"] == "七层动力表"
    assert command["rewrite_source"] == "rule"


def test_memory_rewriter_uses_llm_fallback_when_rules_miss() -> None:
    def fake_llm(message: str):
        assert "冷气" in message
        return {
            "alias_text": "冷气",
            "target_text": "空调",
            "confidence": "medium",
            "source": "llm",
        }

    rewriter = _build_rewriter(llm_rewrite=fake_llm)
    command = rewriter.rewrite_create_command("以后冷气这个叫法你就按空调理解")
    assert command is not None
    assert command["alias_text"] == "冷气"
    assert command["target_text"] == "空调"
    assert command["rewrite_source"] == "llm"
    assert command["rewrite_confidence"] == "medium"
