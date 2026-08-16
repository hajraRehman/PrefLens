from src.parsing import parse_choice


def test_strict_json():
    r = parse_choice('{"choice": "A", "strength": 0.72}')
    assert (r.choice_displayed, r.strength_self_report, r.parse_stage) == ("A", 0.72, "strict_json")


def test_fenced_json():
    r = parse_choice('```json\n{"choice":"B","strength":0.4}\n```')
    assert r.choice_displayed == "B" and r.success


def test_embedded_json_in_prose():
    r = parse_choice('Sure! Here you go: {"choice": "B"} Hope that helps.')
    assert r.choice_displayed == "B" and r.parse_stage == "embedded_json"


def test_strength_percent_scale_is_rescaled():
    assert parse_choice('{"choice":"A","strength":80}').strength_self_report == 0.8


def test_out_of_range_strength_is_absent_not_zero():
    r = parse_choice('{"choice":"A","strength":-3}')
    assert r.choice_displayed == "A" and r.strength_self_report is None


def test_labelled_prose_takes_first_mention():
    r = parse_choice("I would choose A, although B is also reasonable.")
    assert r.choice_displayed == "A" and r.parse_stage == "labelled_text"


def test_bare_token():
    assert parse_choice(" B. ").choice_displayed == "B"


def test_option_prefixed_value():
    assert parse_choice('{"choice": "Option B"}').choice_displayed == "B"


def test_alternate_keys():
    assert parse_choice('{"selection":"A"}').choice_displayed == "A"


def test_failures_are_flagged_not_silently_dropped():
    for bad in ["", "   ", "I genuinely have no preference between these.", "{}"]:
        r = parse_choice(bad)
        assert r.success is False
        assert r.choice_displayed is None
        assert r.parse_stage == "failed"
        assert r.parse_note  # a reason is always recorded


def test_no_letter_preference_in_parser():
    """The parser must not favour 'A' when both letters appear equally."""
    assert parse_choice('{"choice":"B"}').choice_displayed == "B"
    assert parse_choice('{"choice":"A"}').choice_displayed == "A"
