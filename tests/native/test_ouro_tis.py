from types import SimpleNamespace as NS
import pytest
import torch

from slime_plugins.ouro_2a import check_on_policy, policy_loss, rollout_correction


def options(**overrides):
    values=dict(use_tis=True,ouro_rollout_correction='tis',tis_clip_low=0.,tis_clip=2.,
        num_steps_per_rollout=1,use_rollout_logprobs=False,calculate_per_token_loss=False,
        context_parallel_size=1,rollout_temperature=1,rollout_top_p=1,
        eps_clip=.2,eps_clip_high=.2,rollout_max_response_len=10,
        use_kl_loss=False,kl_loss_coef=0.)
    values.update(overrides)
    return NS(**values)


def test_tis_is_detached_old_over_sampler_not_ppo_ratio():
    old=torch.tensor([-.5,-1.,-.7],requires_grad=True)
    rho=torch.tensor([.25,.8,4.])
    sampled=(old.detach()-rho.log()).requires_grad_()
    weights,stats=rollout_correction(options(),old,sampled)
    torch.testing.assert_close(weights,torch.tensor([.25,.8,2.]))
    assert not weights.requires_grad
    assert stats['tis_clip_fraction']==pytest.approx(1/3)
    assert 0<stats['tis_ess_fraction']<=1


def test_tis_corrects_actual_custom_loss_gradient_without_changing_budget(monkeypatch):
    import slime.backends.megatron_utils.loss as native_loss
    monkeypatch.setattr(native_loss,'get_log_probs_and_entropy',lambda logits,**kw:(None,{'log_probs':[logits]}))
    old=torch.tensor([-.5,-1.,-.7],requires_grad=True)
    sampled=(old.detach()-torch.tensor([.25,.8,4.]).log()).requires_grad_()
    current=old.detach().clone().requires_grad_()
    advantage=torch.tensor([2.,-1.,.5])
    batch=dict(log_probs=[old],rollout_log_probs=[sampled],advantages=[advantage],
        loss_masks=[torch.tensor([1.,1.,0.])],unconcat_tokens=[],total_lengths=[],response_lengths=[])
    loss,metrics=policy_loss(options(),batch,current,None)
    loss.backward()
    expected=-advantage*torch.tensor([.25,.8,2.])*batch['loss_masks'][0]/10
    torch.testing.assert_close(current.grad,expected)
    assert old.grad is None and sampled.grad is None
    assert 'tis_clip_fraction' in metrics


def test_correction_and_strict_ieee_gate_are_explicitly_distinct():
    data=dict(log_probs=[torch.tensor([-.5])],rollout_log_probs=[torch.tensor([-.6])])
    check_on_policy(options(),0,data)
    assert data['ouro_logprob_max_error']>.001
    assert data['ouro_tis_ess_fraction']==1
    # slime's rollout logger accepts tensors and sequences, not nested dicts.
    assert all(isinstance(value,(torch.Tensor,list,tuple)) for value in data.values())
    with pytest.raises(ValueError,match='mismatch'):
        check_on_policy(options(use_tis=False),0,data)
    with pytest.raises(ValueError,match='explicitly'):
        check_on_policy(options(ouro_rollout_correction='none'),0,data)


def test_tis_never_hides_nonfinite_or_malformed_data():
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError,match='Invalid'):
            check_on_policy(options(),0,dict(log_probs=[torch.tensor([value])],rollout_log_probs=[torch.tensor([-.6])]))
    with pytest.raises(ValueError,match='Invalid'):
        check_on_policy(options(),0,dict(log_probs=[torch.tensor([-.5,-.6])],rollout_log_probs=[torch.tensor([-.6])]))
