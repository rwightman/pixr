""" Model tests. NOTE: downloads pretrained weights from the HF hub on first run. """
from dataclasses import replace

import pytest
import torch

from pixr.models import (
    ModelArgs, ModelCfg, TextDecoderCfg, ImageEncoderCfg, TextDecoder, EttinTextDecoder, create_model,
    get_model_config, list_models, load_model_state, merge_patch_tokens,
)
from pixr.generation import generate_greedy


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
    if model_type == 'fritter':
        return ModelCfg(
            type='fritter',
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


@pytest.mark.parametrize('model_type', ['cruller', 'fritter'])
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
    dec = EttinTextDecoder(_tiny_cfg('fritter').text_decoder, vocab_size=100, pad_token_id=1).eval()
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



def test_qwen3_decoder_matches_hf():
    import transformers
    from pixr.models import Qwen3TextDecoder
    torch.manual_seed(0)
    cfg = TextDecoderCfg(type='qwen3', name='Qwen/Qwen3-0.6B-Base', num_layers=None, max_length=256, cross_attn_every=0)
    dec = Qwen3TextDecoder(cfg, vocab_size=151936, pad_token_id=151669).eval()
    hf = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B-Base', dtype=torch.float32).eval()
    ids = torch.randint(10, 150000, (2, 40))
    with torch.no_grad():
        a = hf(input_ids=ids).logits
        b = dec(ids)
    assert torch.allclose(a, b, atol=1e-3)
    # w/ (zero init) cross-attn the decoder still matches the pretrained LM, and cache / padding paths are consistent
    xdec = Qwen3TextDecoder(replace(cfg, cross_attn_every=2), vocab_size=151936, pad_token_id=151669).eval()
    enc = torch.randn(2, 30, 1024)
    with torch.no_grad():
        c = xdec(ids, encoder_hidden_states=enc)
    assert torch.allclose(a, c, atol=1e-3)
    for l in xdec.layers:
        if l.cross_attn is not None:
            torch.nn.init.normal_(l.cross_attn.out_proj.weight, std=0.02)
    _check_cache_and_padding(xdec, ids, enc, pad_id=151669, atol=5e-2)


_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _tiny_cronut_cfg(attn_backend='sdpa'):
    from pixr.models import CronutCfg
    return ModelCfg(
        type='cronut',
        image_encoder=ImageEncoderCfg(name='cronut', pretrained=False, image_fmt='L', max_seq_len=64),
        text_decoder=TextDecoderCfg(name='tiktoken:cl100k_base', pretrained=False, max_length=64),
        cronut=CronutCfg(
            width=64, depth=2, num_heads=2, intermediate_size=96, text_embed_dim=48, patch_size=16,
            mrope_section=(6, 5, 5), attn_backend=attn_backend),
    )


def test_cronut_greedy_matches_teacher_forced():
    torch.manual_seed(0)
    model = create_model(_tiny_cronut_cfg(), vocab_size=100, pad_token_id=1).eval()
    patches = _patches()
    prompts = [[5], [6, 7, 8]]
    out = generate_greedy(model, patches, prompt_ids=prompts, stop_token_ids=[2], pad_token_id=1, max_new_tokens=12)
    for i, prompt in enumerate(prompts):
        single = {k: v[i:i + 1] for k, v in patches.items()}
        seq = torch.tensor([prompt + out[i]])
        with torch.no_grad():
            logits = model(single, seq[:, :-1])
        assert logits[0, len(prompt) - 1:].argmax(-1).tolist() == out[i]


def test_cronut_flex_matches_sdpa():
    torch.manual_seed(0)
    ref = create_model(_tiny_cronut_cfg('sdpa'), vocab_size=100, pad_token_id=1).to(_DEV).eval()
    flex = create_model(_tiny_cronut_cfg('flex'), vocab_size=100, pad_token_id=1).to(_DEV).eval()
    flex.load_state_dict(ref.state_dict())
    patches = {k: v.to(_DEV) for k, v in _patches().items()}
    text = torch.randint(3, 100, (2, 9), device=_DEV)
    with torch.no_grad():
        a = ref(patches, text)
        b = flex(patches, text)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_cronut_checkpoint_vocab_expansion(tmp_path):
    torch.manual_seed(0)
    model = create_model(_tiny_cronut_cfg(), vocab_size=100, pad_token_id=1)
    path = tmp_path / 'cronut.pt'
    torch.save({'model': model.state_dict(), 'model_cfg': model.cfg.to_dict()}, path)
    cfg = ModelCfg.from_dict(torch.load(path)['model_cfg'])
    model2 = create_model(cfg, vocab_size=104, pad_token_id=1, pretrained=str(path))
    assert torch.equal(model2.text_embed.weight[:100], model.text_embed.weight)
    assert torch.equal(model2.lm_head.weight[:100], model.lm_head.weight)


@pytest.mark.parametrize('dec_type', ['ettin', 'qwen3'])
def test_prefix_lm_flex_matches_sdpa(dec_type):
    from pixr.models import EttinTextDecoder, Qwen3TextDecoder
    torch.manual_seed(0)
    kw = dict(type=dec_type, pretrained=False, num_layers=3, dim=64, num_heads=4, ffn_dim=96, max_length=64,
              cross_attn_every=0, sliding_window=4, global_attn_every=2, num_kv_heads=2, head_dim=16)
    cls = EttinTextDecoder if dec_type == 'ettin' else Qwen3TextDecoder
    ref = cls(TextDecoderCfg(**kw), vocab_size=100, pad_token_id=1).to(_DEV).eval()
    flex = cls(TextDecoderCfg(**kw, self_attn_backend='flex'), vocab_size=100, pad_token_id=1).to(_DEV).eval()
    flex.load_state_dict(ref.state_dict())
    prefix = torch.randn(2, 10, 64, device=_DEV)
    prefix_mask = torch.ones(2, 10, dtype=torch.bool, device=_DEV)
    prefix_mask[1, 6:] = False
    ids = torch.randint(3, 100, (2, 7), device=_DEV)
    with torch.no_grad():
        a = ref(ids, prefix_embeds=prefix, prefix_mask=prefix_mask)
        b = flex(ids, prefix_embeds=prefix, prefix_mask=prefix_mask)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_tiktoken_tokenizer():
    from pixr.tokenizers import TokenizerCfg, create_tokenizer, get_added_tokens
    tok = create_tokenizer(TokenizerCfg('tiktoken:cl100k_base'), added_tokens=['<s_pretrain>'])
    assert tok.pad_token_id != tok.eos_token_id
    ids = tok('<s_pretrain>Total: $56.78' + tok.eos_token, add_special_tokens=False).input_ids
    assert ids[0] == tok.convert_tokens_to_ids('<s_pretrain>') and ids[-1] == tok.eos_token_id
    assert tok.decode(ids, skip_special_tokens=True) == 'Total: $56.78'
    assert get_added_tokens(tok) == ['<s_pretrain>']


@pytest.mark.parametrize('model_type', ['cruller', 'fritter', 'cronut'])
def test_param_groups_lr_scaled(model_type):
    from pixr.framework.config import OptimizationCfg
    from pixr.framework.trainer import param_groups_lr_scaled
    cfg = _tiny_cronut_cfg() if model_type == 'cronut' else _tiny_cfg(model_type)
    model = create_model(cfg, vocab_size=100, pad_token_id=1)
    layer_map = model.param_layer_map()
    assert set(layer_map) == {n for n, _ in model.named_parameters()}
    opt = OptimizationCfg(
        encoder_lr_scale=0.1, encoder_layer_decay=0.5, bridge_lr_scale=1.0, decoder_lr_scale=0.5,
        decoder_layer_decay=0.5, lr_scale_overrides={r'embed_tokens|text_embed': 4.0})
    groups = param_groups_lr_scaled(model, opt)
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))
    scale_of = {id(p): g['lr_scale'] for g in groups for p in g['params']}
    params = dict(model.named_parameters())
    for name, (component, layer_id, num_layers) in layer_map.items():
        base = {'encoder': 0.1, 'bridge': 1.0, 'decoder': 0.5}[component]
        if component != 'bridge':
            base *= 0.5 ** (num_layers - 1 - layer_id)
        if 'embed_tokens' in name or 'text_embed' in name:
            base *= 4.0
        assert abs(scale_of[id(params[name])] - base) < 1e-9, name
    if model_type == 'cruller':
        assert layer_map['text_decoder.layers.0.encoder_attn.q_proj.weight'][0] == 'bridge'
        assert layer_map['text_decoder.embed_tokens.weight'][1] == layer_map['text_decoder.embed_tokens.weight'][2] - 1
    if model_type == 'fritter':
        assert layer_map['projector.fc1.weight'][0] == 'bridge'
