"""Admission settings for complete, unretracted Ouro generation trajectories."""
import os


def prompt_ids(tokenizer, content):
    # Match slime Dataset's rendered template and _prepare_prompt_ids exactly.
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=content)], tools=None,
        tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def context_capacity(tokenizer, prompts, max_new_tokens, model_limit, kv_capacity):
    lengths = [len(prompt_ids(tokenizer, text)) for text in prompts]
    if not lengths or min(lengths) < 1 or max_new_tokens < 1:
        raise ValueError("Nonempty prompts and a positive generation budget are required")
    required = max(lengths) + max_new_tokens
    # The pinned SGLang worker sets max_req_len = context_len - 1, then
    # scheduler admission subtracts one more slot from the response budget.
    # Merely accepting the HTTP request would silently lose two output tokens.
    reserved = required + 2
    if reserved > min(model_limit, kv_capacity):
        raise ValueError(f"Prompt plus response and scheduler reserve need {reserved} tokens, exceeding model/KV capacity; "
                         "refusing to truncate the dataset or response budget")
    return dict(context_length=min(max(4096, reserved), model_limit, kv_capacity),
                max_prompt_tokens=max(lengths), required_tokens=required,
                scheduler_reserve_tokens=2,
                prompt_count=len(lengths), max_new_tokens=max_new_tokens)


def token_exit_reservation(max_new_tokens):
    if max_new_tokens < 1:
        raise ValueError("Generation budget must be positive")
    # SGLang's ratio normally decays to 0.098 and permits KV overcommit.
    # Dense re-prefill cannot reconstruct the last-exited KV of a gated prefix.
    return dict(SGLANG_INIT_NEW_TOKEN_RATIO="1",
                SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR="1",
                SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=str(max(4096, max_new_tokens)))


def configure_token_exit_serving(max_new_tokens):
    settings = token_exit_reservation(max_new_tokens)
    os.environ.update(settings)
    return settings
