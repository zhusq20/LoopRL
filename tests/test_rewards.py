from looprl.rewards import last_boxed, score_answer


def test_nested_and_last_box():
    assert last_boxed(r"old \boxed{1} final \boxed{\frac{2}{3}}") == r"\frac{2}{3}"
    assert last_boxed("no box") is None


def test_equivalence_and_incomplete_thinking():
    assert score_answer(r"\boxed{0.5}", r"\frac{1}{2}")["reward"] == 1
    assert score_answer(r"\boxed{2}", "3")["reward"] == 0
    assert score_answer(r"<think>maybe \boxed{2}", "2")["reward"] == 0
    assert score_answer(r"<think>\boxed{2}</think>\boxed{3}", "2")["reward"] == 0
