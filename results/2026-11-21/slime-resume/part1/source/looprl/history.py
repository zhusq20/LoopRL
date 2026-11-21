"""M7: full-vocabulary KL between dense and ragged histories on a fixed text."""
import json

import torch

from .data import write_json
from .experiments import new_run, selected_rows, seed_everything
from .ouro import OuroAdapter


def history_probe(args):
    path = new_run(args)
    seed_everything(args.seed)
    adapter = OuroAdapter.load(args.model, args.device)
    rows = selected_rows(args)
    samples = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            prompt = adapter.prompt_ids(row["problem"])
            rollout = adapter.generate(prompt, 4, args.max_new_tokens, seed=args.seed + i)
            tokens = rollout.response_ids
            _, lq, dense_mix = adapter.ragged_replay(prompt, tokens, [4] * len(tokens), return_mixture=True)
            gen = torch.Generator(device=adapter.device).manual_seed(args.seed + i)
            sampled = (torch.multinomial(lq[0].exp(), 1, generator=gen).squeeze(-1) + 1).tolist()
            jumps = [4 if j % 2 == 0 else 1 for j in range(len(tokens))]
            for mode, exits in [("native_gate_sample", sampled), ("alternating_4_1", jumps)]:
                _, _, ragged_mix = adapter.ragged_replay(prompt, tokens, exits, return_mixture=True)
                kl = (ragged_mix.exp() * (ragged_mix - dense_mix)).sum(-1).clamp_min(0)
                record = {"id": row["id"], "mode": mode, "tokens": tokens, "exits": exits,
                          "kl_ragged_to_dense": kl.tolist(), "mean_kl": float(kl.mean()),
                          "max_kl": float(kl.max()), "groups": {}}
                for previous_exit in (1, 2, 3, 4):
                    for jumped in (False, True):
                        # At j, only previous queries' exits influence history;
                        # e[0] does not truncate the dense prompt boundary.
                        positions = [j for j in range(2, len(tokens))
                                     if exits[j - 1] == previous_exit and (exits[j - 1] != exits[j - 2]) == jumped]
                        if positions:
                            record["groups"][f"previous_exit={previous_exit},jump={jumped}"] = {
                                "positions": len(positions), "mean_kl": float(kl[positions].mean())}
                samples.append(record)
                write_json(path / "m7.json", {"samples": samples,
                    "scope": "Short fixed-text KV sensitivity probe; not an accuracy or policy improvement result."})
                print(json.dumps({"id": row["id"], "mode": mode, "mean_kl": record["mean_kl"], "max_kl": record["max_kl"]}), flush=True)
    write_json(path / "status.json", {"status": "completed", "samples": len(samples)})
    return {"samples": len(samples), "result": str(path / "m7.json")}
