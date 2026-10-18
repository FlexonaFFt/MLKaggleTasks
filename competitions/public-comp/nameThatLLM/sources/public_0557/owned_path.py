"""Minimal byte-preserved scientific functions extracted from the reviewed owned full-fit path."""


def load_owned_base(base_dir, loading, deadline):
    import torch
    from transformers import AutoModel
    def load_base():
        deadline()
        base, info = AutoModel.from_pretrained(base_dir, dtype=torch.float16, device_map={"": 0},
            attn_implementation="sdpa", trust_remote_code=False, local_files_only=True, output_loading_info=True)
        assert not info.get("missing_keys") and not info.get("mismatched_keys") and not info.get("error_msgs"), info
        assert set(info.get("unexpected_keys", [])) <= {"lm_head.weight"}, info
        assert len(base.layers) == 36 and not hasattr(base, "lm_head")
        assert all(parameter.device.type == "cuda" and parameter.device.index == 0 for parameter in base.parameters())
        base.requires_grad_(False)
        loading.append(info)
        return base
    return load_base()


def make_representation(model, tokenizer, tokens, layers, norms):
    import torch
    def representation(rows):
        batch = tokenizer.pad({"input_ids": [tokens[i] for i in rows]}, return_tensors="pt").to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        assert len(hidden) == 37
        pieces = []
        with torch.autocast("cuda", enabled=False):
            for layer, norm in zip(layers, norms):
                mask = batch.attention_mask.unsqueeze(-1).float()
                mask[:, 0] = 0
                assert bool((mask.sum(1) > 0).all())
                mean = (hidden[layer].float() * mask).sum(1) / mask.sum(1)
                pieces.append(norm(mean))
            z = torch.cat(pieces, -1)
        assert bool(torch.isfinite(z).all())
        return z, int(batch.attention_mask.sum()), int(batch.input_ids.numel())
    return representation


def tokenize_owned(base_dir, texts, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True, trust_remote_code=False)
    assert tokenizer.bos_token_id is None and tokenizer.eos_token_id == 128001
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 128004
    tokens = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=447)["input_ids"]
    tokens = [row + [128001] for row in tokens]
    assert all(2 <= len(row) <= 448 for row in tokens)
    return tokenizer, tokens