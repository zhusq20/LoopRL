"""Conservative boxed-answer extraction and math-verify scoring."""
import re


def last_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(matches):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
    return None


def score_answer(response: str, gold: str) -> dict:
    # Reasoning-internal candidate answers must not count as final answers.
    if "<think>" in response and "</think>" not in response:
        return {"reward": 0., "answer": None, "reason": "unfinished_thinking"}
    final = response.rsplit("</think>", 1)[-1]
    answer = last_boxed(final)
    if answer is None:
        return {"reward": 0., "answer": None, "reason": "no_boxed_answer"}
    from math_verify import parse, verify, LatexExtractionConfig
    target = last_boxed(gold) or gold.strip()
    parsed_gold = parse("$" + target + "$", extraction_config=[LatexExtractionConfig()], extraction_mode="first_match")
    parsed_answer = parse("$" + answer + "$", extraction_config=[LatexExtractionConfig()], extraction_mode="first_match")
    correct = bool(parsed_gold and parsed_answer and verify(parsed_gold, parsed_answer))
    return {"reward": float(correct), "answer": answer, "reason": "equivalent" if correct else "mismatch"}
