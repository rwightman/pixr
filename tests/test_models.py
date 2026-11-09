""" Model tests. NOTE: downloads pretrained weights from the HF hub on first run. """
import pytest
import torch

from pixparse.models import (
    ModelArgs, ModelCfg, TextDecoderCfg, ImageEncoderCfg, TextDecoder, EttinTextDecoder, create_model,
    get_model_config, list_models, load_model_state, merge_patch_tokens,
)
from pixparse.generation import generate_greedy


def _hf_bart_causal_lm(name):
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    c = transformers.AutoConfig.from_pretrained(name)
    c.is_decoder, c.add_cross_attention, c.is_encoder_decoder = True, True, False
    hf = transformers.BartForCausalLM.from_pretrained(name, config=c).eval()
    # NOTE transformers 5.x does not tie / load `shared.weight` into the decoder embeddings for bart-base
    sd = load_file(hf_hub_download(name, 'model.safetensors'))
    with torch.no_grad():
        hf.model.decoder.embed_tokens.weight.copy_(sd['shared.weight'])
    hf.lm_head.weight = hf.model.decoder.embed_tokens.weight
    return hf


def _check_cache_and_padding(dec, ids, enc, pad_id, atol):
    with torch.no_grad():
        full = dec(ids, encoder_hidden_states=enc)
        cross = dec.project_cross_kv(enc)
        cache = [{} for _ in dec.layers]
        split = ids.shape[1] // 2
        out = [dec(ids[:, :split], cross_kv=cross, cache=cache)]
        for t in range(split, ids.shape[1]):
            out.append(dec(ids[:, t:t + 1], cross_kv=cross, cache=cache))
        cached = torch.cat(out, 1)
        pad = 5
        ids_p = torch.cat([torch.full((ids.shape[0], pad), pad_id), ids], 1)
        am = torch.cat([torch.zeros(ids.shape[0], pad, dtype=torch.bool), torch.ones_like(ids, dtype=torch.bool)], 1)
        pos = (am.long().cumsum(1) - 1).clamp(min=0)
        padded = dec(ids_p, encoder_hidden_states=enc, position_ids=pos, attention_mask=am)[:, pad:]
    assert torch.allclose(cached, full, atol=atol)
    assert not torch.isnan(padded).any()
    assert torch.allclose(padded, full, atol=atol)


def test_bart_decoder_matches_hf():
    torch.manual_seed(0)
    cfg = TextDecoderCfg(name='facebook/bart-base', num_layers=6, reinit_cross_attn=False)
    dec = TextDecoder(cfg, vocab_size=50265).eval()
    hf = _hf_bart_causal_lm('facebook/bart-base')
    ids = torch.randint(4, 50000, (2, 33))
    enc = torch.randn(2, 40, 768)
    with torch.no_grad():
        a = hf(input_ids=ids, encoder_hidden_states=enc).logits
        b = dec(ids, encoder_hidden_states=enc)
    assert torch.allclose(a, b, atol=1e-4)
    _check_cache_and_padding(dec, ids, enc, pad_id=1, atol=1e-2)


def test_bart_decoder_reinit_cross_attn_starts_as_lm():
    torch.manual_seed(0)
    dec = TextDecoder(TextDecoderCfg(name='facebook/bart-base', num_layers=6), vocab_size=50265).eval()
    ids = torch.randint(4, 50000, (2, 17))
    with torch.no_grad():
        a = dec(ids, encoder_hidden_states=torch.randn(2, 20, 768))
        b = dec(ids, encoder_hidden_states=torch.randn(2, 30, 768))
    assert torch.allclose(a, b)  # zero init cross-attn output, independent of encoder at init


def test_ettin_decoder_matches_hf():
    import transformers
    torch.manual_seed(0)
    cfg = TextDecoderCfg(type='ettin', name='jhu-clsp/ettin-decoder-150m', num_layers=None, max_length=1024)
    dec = EttinTextDecoder(cfg, vocab_size=50368, pad_token_id=50283).eval()
    hf = transformers.AutoModelForCausalLM.from_pretrained('jhu-clsp/ettin-decoder-150m').eval()
    ids = torch.randint(10, 50000, (2, 150))  # > sliding window
    enc = torch.randn(2, 40, 768)
    with torch.no_grad():
        a = hf(input_ids=ids).logits
        b = dec(ids, encoder_hidden_states=enc)
    assert torch.allclose(a, b, atol=1e-3)
    _check_cache_and_padding(dec, ids, enc, pad_id=50283, atol=1e-2)


def test_model_configs_and_overrides():
    names = list_models()
    assert 'cruller_naflex_clip_base' in names
    args = ModelArgs(name='cruller_naflex_clip_base', max_seq_len=1234, text_max_length=256, pretrained=False)
    cfg = args.resolve()
    assert cfg.image_encoder.max_seq_len == 1234
    assert cfg.text_decoder.max_length == 256
    assert not cfg.image_encoder.pretrained and not cfg.text_decoder.pretrained
    assert get_model_config('cruller_naflex_clip_base').image_encoder.max_seq_len == 4096  # registry unchanged


def _tiny_cfg(model_type='cruller'):
    image_encoder = ImageEncoderCfg(
        name='naflexvit', pretrained=False, image_fmt='L', max_seq_len=64,
        model_kwargs=dict(patch_size=16, embed_dim=64, depth=2, num_heads=2, class_token=True, pre_norm=True),
    )
    if model_type == 'eclair':
        return ModelCfg(
            type='eclair',
            image_encoder=image_encoder,
            text_decoder=TextDecoderCfg(
                type='ettin', pretrained=False, num_layers=3, dim=64, num_heads=2, ffn_dim=96, max_length=64,
                sliding_window=4, global_attn_every=2, cross_attn_every=0),
            prefix_token_merge=2,
        )
    return ModelCfg(
        image_encoder=image_encoder,
        text_decoder=TextDecoderCfg(pretrained=False, num_layers=2, dim=64, num_heads=2, ffn_dim=128, max_length=64),
    )


def _patches(B=2, H=4, W=3, num_invalid=4):
    coord = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1).reshape(1, H * W, 2)
    valid = torch.ones(B, H * W, dtype=torch.bool)
    valid[1, H * W - num_invalid:] = False
    return {'patches': torch.randn(B, H * W, 256), 'patch_coord': coord.expand(B, -1, -1), 'patch_valid': valid}


def test_merge_patch_tokens():
    H, W, C = 4, 6, 3
    coord = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), -1).reshape(1, H * W, 2)
    tokens = torch.arange(H * W * C, dtype=torch.float).reshape(1, H * W, C)
    merged, valid = merge_patch_tokens(tokens, torch.ones(1, H * W, dtype=torch.bool), coord, 2)
    assert merged.shape == (1, 6, 12) and valid.all()
    # reference: pixel unshuffle of the (H, W, C) grid
    grid = tokens.reshape(H, W, C)
    ref = grid.reshape(H // 2, 2, W // 2, 2, C).permute(0, 2, 1, 3, 4).reshape(1, 6, 4 * C)
    assert torch.equal(merged, ref)


@pytest.mark.parametrize('model_type', ['cruller', 'eclair'])
def test_greedy_matches_teacher_forced(model_type):
    torch.manual_seed(0)
    model = create_model(_tiny_cfg(model_type), vocab_size=100, pad_token_id=1).eval()
    patches = _patches()
    prompts = [[5], [6, 7, 8]]
    out = generate_greedy(model, patches, prompt_ids=prompts, stop_token_ids=[2], pad_token_id=1, max_new_tokens=12)
    assert len(out) == 2 and all(1 <= len(o) <= 12 for o in out)
    for i, prompt in enumerate(prompts):
        single = {k: v[i:i + 1] for k, v in patches.items()}
        seq = torch.tensor([prompt + out[i]])
        with torch.no_grad():
            logits = model(single, seq[:, :-1])
        assert logits[0, len(prompt) - 1:].argmax(-1).tolist() == out[i]


def test_ettin_prefix_lm_mask():
    """ Prefix tokens attend bidirectionally (changing a later prefix token changes earlier prefix outputs is
    not observable from text logits, so check text logits depend on all prefix tokens and causality of text). """
    torch.manual_seed(0)
    dec = EttinTextDecoder(_tiny_cfg('eclair').text_decoder, vocab_size=100, pad_token_id=1).eval()
    prefix = torch.randn(1, 6, 64)
    ids = torch.randint(3, 100, (1, 5))
    with torch.no_grad():
        a = dec(ids, prefix_embeds=prefix, prefix_mask=torch.ones(1, 6, dtype=torch.bool))
        ids2 = ids.clone(); ids2[0, -1] = (ids2[0, -1] + 1) % 100
        b = dec(ids2, prefix_embeds=prefix, prefix_mask=torch.ones(1, 6, dtype=torch.bool))
        # masking a prefix token (invalid) == removing it
        m = torch.ones(1, 6, dtype=torch.bool); m[0, 2] = False
        c = dec(ids, prefix_embeds=prefix, prefix_mask=m)
        d = dec(ids, prefix_embeds=torch.cat([prefix[:, :2], prefix[:, 3:]], 1), prefix_mask=torch.ones(1, 5, dtype=torch.bool))
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)  # text is causal
    assert torch.allclose(c, d, atol=1e-4)  # padded prefix tokens are ignored, positions contiguous


def test_checkpoint_vocab_expansion(tmp_path):
    torch.manual_seed(0)
    model = create_model(_tiny_cfg(), vocab_size=100, pad_token_id=1).eval()
    path = tmp_path / 'ckpt.pt'
    torch.save({'model': model.state_dict(), 'model_cfg': model.cfg.to_dict()}, path)

    cfg = ModelCfg.from_dict(torch.load(path)['model_cfg'])
    model2 = create_model(cfg, vocab_size=105, pad_token_id=1, pretrained=str(path)).eval()
    w1 = model.text_decoder.embed_tokens.weight
    w2 = model2.text_decoder.embed_tokens.weight
    assert torch.equal(w1, w2[:100])
    assert torch.allclose(w2[100:], w1.mean(0, keepdim=True).expand(5, -1))

