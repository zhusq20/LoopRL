"""Real-checkpoint acceptance checks before any RL update."""
import copy
import json

import torch

from .data import write_json
from .experiments import new_run, seed_everything
from .objectives import mixture_terms
from .ouro import OuroAdapter


def preflight(args):
    path = new_run(args)
    seed_everything(args.seed)
    adapter = OuroAdapter.load(args.model, args.device)
    prompt = adapter.prompt_ids("What is 6 times 7?")
    report = {"model": args.model, "parameters": sum(p.numel() for p in adapter.model.parameters()),
              "fixed": {}, "ragged": {}}
    with torch.no_grad():
        for depth in (1, 2, 3, 4):
            count = [0]
            handle = adapter.model.model.layers[0].register_forward_hook(lambda *unused: count.__setitem__(0, count[0] + 1))
            ids = torch.tensor([prompt], device=adapter.device)
            _, hs, _ = adapter.forward(ids, depth)
            handle.remove()
            assert count[0] == depth, f"depth {depth} secretly executed {count[0]} loops"
            _, full, _ = adapter.forward(ids, 4)
            readout_error = float((hs[-1] - full[depth - 1]).abs().max())
            assert readout_error == 0, readout_error
            rollout = adapter.generate(prompt, depth, args.tokens, seed=args.seed)
            lp = adapter.response_log_probs(rollout)
            error = float((lp - torch.tensor(rollout.log_probs, device=adapter.device)).abs().max())
            report["fixed"][str(depth)] = {"executed_loops": count[0], "hidden_readout_error": readout_error,
                                           "rollout_update_logp_max_error": error, "tokens": len(rollout.response_ids)}
            assert error <= args.logprob_tolerance, report["fixed"][str(depth)]
            print(json.dumps({"fixed": depth, **report["fixed"][str(depth)]}), flush=True)
        for shift in (0., 2.):
            rollout = adapter.generate_ragged(prompt, args.tokens, gate_shift=shift, seed=args.seed)
            lp, lq = adapter.ragged_replay(prompt, rollout.response_ids, rollout.depths, gate_shift=shift)
            e = torch.tensor([rollout.depths], device=adapter.device)
            mix, posterior, _ = mixture_terms(lp, lq, e)
            error = float((mix + posterior - torch.tensor([rollout.log_probs], device=adapter.device)).abs().max())
            report["ragged"][str(shift)] = {"exits": rollout.depths, "joint_logp_max_error": error}
            assert error <= args.logprob_tolerance, report["ragged"][str(shift)]
            print(json.dumps({"shift": shift, **report["ragged"][str(shift)]}), flush=True)

    # A tiny instance of the SAME Ouro implementation verifies differentiable
    # ragged replay and equality to dense teacher forcing when every e=4.
    config = copy.deepcopy(adapter.model.config)
    config.hidden_size, config.intermediate_size, config.head_dim = 32, 64, 16
    config.num_hidden_layers = 2
    config.layer_types = ["full_attention"] * 2
    config.num_attention_heads = config.num_key_value_heads = 2
    config.vocab_size = 32
    config.torch_dtype = torch.float32
    tiny_model = type(adapter.model)(config).to(adapter.device).float().eval()
    tiny = OuroAdapter(tiny_model, adapter.tokenizer)
    prefix, response = [3, 5, 7], [2, 9, 4, 6]
    ids = torch.tensor([prefix + response[:-1]], device=tiny.device)
    _, hs, _ = tiny.forward(ids, 4)
    dense = torch.stack([tiny.model.lm_head(h[0, len(prefix) - 1:]).log_softmax(-1).gather(-1,
                        torch.tensor(response, device=tiny.device)[:, None]).squeeze(-1) for h in hs], -1)
    replay, _ = tiny.ragged_replay(prefix, response, [4] * 4)
    dense_error = float((dense - replay[0]).abs().max())
    torch.testing.assert_close(dense, replay[0], atol=1e-5, rtol=1e-5)
    lp, _ = tiny.ragged_replay(prefix, response, [4, 1, 4, 2])
    grad, = torch.autograd.grad(lp[0, -1].sum(), tiny.model.model.embed_tokens.weight)
    prefix_grad = float(grad[prefix[0]].norm())
    assert prefix_grad > 0, "Gradient was cut through reused historical KV"
    report["tiny_ouro"] = {"dense_ragged_logp_max_error": dense_error, "prefix_embedding_gradient_norm": prefix_grad}
    report["passed"] = True
    write_json(path / "validation.json", report)
    write_json(path / "status.json", {"status": "completed"})
    return report
