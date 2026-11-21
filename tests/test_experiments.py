from looprl.experiments import summarize


def row(depth, reward, question="q", repeat=0):
    return {"depth": depth, "reward": reward, "id": question, "repeat": repeat,
            "response_ids": [7, 8], "depths": [depth, depth], "terminated": True,
            "prefill_loop_tokens": depth * 3}


def test_oracle_uses_only_paired_depths_2_3_4_and_excludes_loop1():
    records = [row(1, 1), row(2, 0), row(3, 0), row(4, 0), row(2, 1, question="unpaired")]
    summary = summarize(records)
    assert summary["M4"]["paired_samples"] == 1
    assert summary["M4"]["oracle_accuracy"] == 0
    assert summary["M4"]["oracle_minus_4"] == 0
    assert summary["by_depth"]["4"]["mean_executed_total_loop_tokens"] == 16
    assert summary["by_depth"]["4"]["repeat_accuracy_std"] is None


def test_summary_reports_sampling_variation_and_paired_overlooping():
    records = [row(2, 1, repeat=r) for r in (0, 1)] + [row(3, 1, repeat=r) for r in (0, 1)]
    records += [row(4, 0, repeat=0), row(4, 1, repeat=1)]
    summary = summarize(records)
    assert summary["M4"]["oracle_minus_4"] == .5
    assert summary["M4"]["shallow_correct_deep_wrong"] == .5
    assert summary["by_depth"]["4"]["repeat_accuracy_std"] > 0
