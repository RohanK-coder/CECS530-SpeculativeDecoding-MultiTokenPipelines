from fresh_specdecode.decoders.speculative import SpeculativeDecoder


def test_verify_all_accept():
    result = SpeculativeDecoder.verify_proposed_tokens([1, 2, 3], [1, 2, 3])
    assert result.accepted_prefix_len == 3
    assert result.fallback_token is None
    assert result.accepted_tokens == [1, 2, 3]
    assert result.rejected_count == 0


def test_verify_prefix_then_reject():
    result = SpeculativeDecoder.verify_proposed_tokens([1, 9, 3], [1, 2, 3])
    assert result.accepted_prefix_len == 1
    assert result.fallback_token == 9
    assert result.accepted_tokens == [1]
    assert result.rejected_count == 2


def test_verify_reject_first_token():
    result = SpeculativeDecoder.verify_proposed_tokens([8, 2, 3], [1, 2, 3])
    assert result.accepted_prefix_len == 0
    assert result.fallback_token == 8
    assert result.accepted_tokens == []
    assert result.rejected_count == 3
