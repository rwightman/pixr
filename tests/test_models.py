""" Model tests. NOTE: downloads pretrained weights from the HF hub on first run. """
import copy
from dataclasses import replace

import math

import pytest
import torch

from pixr.models import (
    ModelArgs, ModelCfg, TextDecoderCfg, ImageEncoderCfg, TextDecoder, EttinTextDecoder, create_model,
    get_model_config, list_models, load_model_state, merge_patch_tokens, LayerKVCache,
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
        memory = dec.prepare_cross(enc)
        cache = [LayerKVCache() for _ in dec.layers]
        split = ids.shape[1] // 2
        out = [dec(ids[:, :split], cross_memory=memory, cache=cache)]
        assert memory.states is None  # K / V projected into the per layer slots at prefill, states released
        for t in range(split, ids.shape[1]):
            out.append(dec(ids[:, t:t + 1], cross_memory=memory, cache=cache))
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


@pytest.mark.parametrize('model_type,dec_type', [
    ('cruller', 'bart'), ('cruller', 'ettin'), ('cruller', 'qwen3'),
    ('fritter', 'ettin'), ('fritter', 'qwen3'), ('cronut', None),
])
@pytest.mark.parametrize('precision', ['fp32', 'bf16', 'fsdp_bf16'])
def test_generation_cache_growth_and_logits(model_type, dec_type, precision):
    """ Cached decoding matches a full forward, including when FSDP rebuilds input containers. """
    torch.manual_seed(4)
    cfg = _tiny_cronut_cfg() if model_type == 'cronut' else _tiny_cfg(model_type)
    if dec_type is not None:
        cfg = replace(cfg, text_decoder=replace(
            cfg.text_decoder, type=dec_type, dropout=0., num_kv_heads=1, head_dim=32))
    dtype = torch.float32 if precision == 'fp32' else torch.bfloat16
    ref = create_model(cfg, vocab_size=100, pad_token_id=1).eval().to(dtype)
    model = copy.deepcopy(ref)
    if precision == 'fsdp_bf16':
        from functools import partial
        from torch.distributed.utils import _apply_to_tensors
        from torch.distributed.fsdp._fully_shard._fsdp_common import _cast_fp_tensor

        # Exercise FSDP's actual input casting on CPU, at every decoder block boundary.
        cast = partial(_cast_fp_tensor, dtype)

        def cast_inputs(module, args, kwargs):
            return _apply_to_tensors(cast, args), _apply_to_tensors(cast, kwargs)

        layers = model.layers if model_type == 'cronut' else model.text_decoder.layers
        for layer in layers:
            layer.register_forward_pre_hook(cast_inputs, with_kwargs=True)

    patches = _patches()
    patches['patches'] = patches['patches'].to(dtype)
    ids = torch.tensor([[1, 1, 5], [6, 7, 8]])
    mask = ids != 1
    tolerance = 1e-5 if dtype == torch.float32 else 5e-3
    with torch.no_grad():
        logits, state = model.prefill(model.encode_image(patches), ids, mask)
        caches = tuple(state.cache)
        prefix_len = state.prefix_len
        for step in range(4):
            for original, cache in zip(caches, state.cache):
                assert cache is original
                assert cache.k is not None and cache.v is not None
                assert cache.k.shape[2] == cache.v.shape[2] == prefix_len + ids.shape[1]
                assert cache.k.dtype == cache.v.dtype == dtype
                assert cache.k.device == cache.v.device == patches['patches'].device
            # Compare each sample with an unpadded full forward; the batch has padded images and prompts.
            for i in range(ids.shape[0]):
                single = {k: v[i:i + 1] for k, v in patches.items()}
                expected = ref(single, ids[i:i + 1, mask[i]])[:, -1]
                torch.testing.assert_close(logits[i:i + 1], expected, atol=tolerance, rtol=tolerance)
            if step < 3:
                token = torch.tensor([[11 + step], [14 + step]])
                ids = torch.cat([ids, token], dim=1)
                mask = torch.cat([mask, torch.ones_like(token, dtype=torch.bool)], dim=1)
                if precision == 'fsdp_bf16':
                    # The root step method casts its state argument as well as the layer inputs.
                    state = _apply_to_tensors(cast, state)
                logits, state = model.step(token, state)


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


@pytest.mark.parametrize('model_type', ['cruller', 'fritter', 'cronut'])
def test_no_padding_patch_valid_none(model_type):
    """ A batch w/o padding gives the same result w/ the validity mask dropped (patch_valid None). """
    from pixr.task.common import drop_full_patch_valid
    torch.manual_seed(0)
    cfg = _tiny_cronut_cfg() if model_type == 'cronut' else _tiny_cfg(model_type)
    model = create_model(cfg, vocab_size=100, pad_token_id=1).eval()
    patches = _patches(num_invalid=0)
    dropped = drop_full_patch_valid(patches)
    assert dropped['patch_valid'] is None and patches['patch_valid'] is not None  # input dict not modified
    padded = _patches()
    assert drop_full_patch_valid(padded) is padded  # masks w/ padding are kept
    text = torch.tensor([[5, 6, 7, 8], [5, 9, 10, 11]])
    with torch.no_grad():
        ref, out = model(patches, text), model(dropped, text)
    assert torch.allclose(ref, out, atol=1e-5)
    prompts = [[5], [6, 7, 8]]
    kw = dict(prompt_ids=prompts, stop_token_ids=[2], pad_token_id=1, max_new_tokens=8)
    assert generate_greedy(model, patches, **kw) == generate_greedy(model, dropped, **kw)


def test_trim_padding_rounds_to_multiple():
    from pixr.task.common import trim_padding
    text = torch.ones(3, 200, dtype=torch.long)  # 1 == pad
    text[0, :70], text[1, :5] = 7, 7
    target = text.clone()
    assert trim_padding(text, target, 1)[0].shape == (3, 70)
    a, b = trim_padding(text, target, 1, multiple=64)
    assert a.shape == b.shape == (3, 128) and (a[0, 70:] == 1).all()
    text[2, :199] = 7
    assert trim_padding(text, target, 1, multiple=64)[0].shape == (3, 200)  # capped at the padded length


@pytest.mark.parametrize('interleaved', [True, False])
def test_mrope_uses_spatial_positions(interleaved):
    from pixr.models.cronut import MRoPE
    rope = MRoPE(64, (16, 8, 8) if not interleaved else (11, 11, 10), 1e4, interleaved=interleaved)
    pos = torch.stack([torch.zeros(1, 6, dtype=torch.long), torch.arange(6)[None], torch.arange(6)[None] * 2])
    cos = rope(pos)[0]
    for axis in (1, 2):
        moved = pos.clone()
        moved[axis] += 3
        assert not torch.equal(rope(moved)[0], cos), f'axis {axis} positions have no effect'
    if not interleaved:
        # contiguous T / H / W frequency sections (Qwen2-VL)
        freqs = pos[..., None].float() * rope.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        ref = torch.cat([c[i % 3] for i, c in enumerate(emb.split([16, 8, 8] * 2, dim=-1))], dim=-1).cos()
        torch.testing.assert_close(cos[:, 0], ref)


# --- generation interface: per image memory, reorderable sequence state, beam search ---

def _gen_model(kind):
    """ tiny models covering every family / decoder: cruller (bart, ettin, qwen3 cross-attn), fritter, cronut """
    torch.manual_seed(0)
    if kind == 'cronut':
        cfg = _tiny_cronut_cfg()
    elif kind in ('cruller_ettin', 'cruller_qwen3'):
        cfg = replace(_tiny_cfg(), text_decoder=TextDecoderCfg(
            type=kind.split('_')[1], pretrained=False, num_layers=3, dim=64, num_heads=4, ffn_dim=96, max_length=64,
            cross_attn_every=1, sliding_window=4, global_attn_every=2, num_kv_heads=2, head_dim=16))
    else:
        cfg = _tiny_cfg(kind)
    model = create_model(cfg, vocab_size=100, pad_token_id=1).eval()
    # zero-init cross-attn output projections would make generation independent of the image
    for n, p in model.named_parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    return model


_GEN_MODELS = ['cruller', 'cruller_ettin', 'cruller_qwen3', 'fritter', 'cronut']


def _teacher_forced_last(model, patches, i, seq):
    """ next token logits for sample i's image after `seq` (prompt + generated), full forward w/o caches """
    single = {k: v[i:i + 1] for k, v in patches.items()}
    with torch.no_grad():
        return model(single, torch.tensor([seq]))[0, -1]


@pytest.mark.parametrize('kind', ['cruller', 'cruller_ettin', 'cruller_qwen3'])
def test_cross_kv_projected_once_inside_layer(kind):
    model = _gen_model(kind)
    calls = []
    for i, layer in enumerate(model.text_decoder.layers):
        attn = getattr(layer, 'encoder_attn', None) or getattr(layer, 'cross_attn', None)
        if attn is None:
            continue
        orig = attn.project_kv
        attn.project_kv = lambda x, _orig=orig, _i=i: (calls.append(_i), _orig(x))[1]
    patches = _patches()
    with torch.no_grad():
        memory = model.encode_image(patches)
        assert not calls and all(s.k is None for s in memory.slots)  # nothing projected outside the layers
        ids = torch.tensor([[5, 6], [7, 8]])
        logits, state = model.prefill(memory, ids, torch.ones_like(ids, dtype=torch.bool))
        assert state.memory is memory and memory.states is None
        slots = tuple(memory.slots)
        for _ in range(4):
            logits, state = model.step(logits.argmax(-1)[:, None], state)
    num_cross = len(calls)
    assert num_cross > 0 and sorted(calls) == sorted(set(calls))  # each cross-attn layer projected exactly once
    assert all(a is b for a, b in zip(slots, state.memory.slots))


@pytest.mark.parametrize('kind', _GEN_MODELS)
def test_state_reorder_children_independent(kind):
    """ Hypotheses duplicated from one parent decode independently, matching teacher forced forwards. """
    model = _gen_model(kind)
    patches = _patches()  # 2 images, the 2nd w/ padding
    prompts = [[5, 6], [7, 8]]
    ids = torch.tensor(prompts)
    with torch.no_grad():
        memory = model.encode_image(patches)
        logits, state = model.prefill(memory, ids, torch.ones_like(ids, dtype=torch.bool))
        # expand to 2 hypotheses per image (image-major), w/ different continuations
        state.reorder(torch.tensor([0, 0, 1, 1]))
        image_of = [0, 0, 1, 1]
        seqs = [list(prompts[i]) for i in image_of]
        for step_tokens, parents in (
                ([11, 12, 13, 14], None),
                ([21, 22, 23, 24], [1, 1, 2, 3]),  # repeated parent within image 0, identity for image 1
                ([31, 32, 33, 34], [0, 1, 3, 3]),  # repeated parent within image 1
        ):
            if parents is not None:
                state.reorder(torch.tensor(parents))
                seqs = [list(seqs[p]) for p in parents]
            seqs = [s + [t] for s, t in zip(seqs, step_tokens)]
            logits, state = model.step(torch.tensor(step_tokens)[:, None], state)
            for row in range(4):
                expected = _teacher_forced_last(model, patches, image_of[row], seqs[row])
                torch.testing.assert_close(logits[row], expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('kind', ['cruller', 'cruller_ettin', 'cruller_qwen3'])
def test_cruller_several_prompts_per_image(kind):
    """ One image memory shared by several prompts (image-major), cross K / V are not expanded. """
    model = _gen_model(kind)
    patches = _patches()
    ids = torch.tensor([[5, 6], [9, 10], [7, 8], [11, 12]])  # 2 prompts for each of the 2 images
    with torch.no_grad():
        memory = model.encode_image(patches)
        logits, state = model.prefill(memory, ids, torch.ones_like(ids, dtype=torch.bool))
        assert all(s.k is None or s.k.shape[0] in (1, 2) for s in memory.slots)  # per image (1 if packed)
        for row in range(4):
            expected = _teacher_forced_last(model, patches, row // 2, ids[row].tolist())
            torch.testing.assert_close(logits[row], expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize('kind', _GEN_MODELS)
def test_beam_search(kind):
    from pixr.generation import generate_beam
    model = _gen_model(kind)
    patches = _patches()
    prompts = [[5], [6, 7, 8]]
    kw = dict(prompt_ids=prompts, stop_token_ids=[2], pad_token_id=1, max_new_tokens=10)

    # one beam is greedy decoding
    greedy = generate_greedy(model, patches, **kw)
    assert generate_beam(model, patches, num_beams=1, **kw) == greedy

    # scores are the sum of the returned sequence's token log probs, checked against teacher forced forwards
    out, scores = generate_beam(model, patches, num_beams=4, return_scores=True, **kw)
    for i, (prompt, seq, score) in enumerate(zip(prompts, out, scores)):
        assert 1 <= len(seq) <= 10
        single = {k: v[i:i + 1] for k, v in patches.items()}
        with torch.no_grad():
            logits = model(single, torch.tensor([prompt + seq[:-1]]))[0, len(prompt) - 1:]
        log_probs = logits.float().log_softmax(-1)
        expected = log_probs[torch.arange(len(seq)), torch.tensor(seq)].sum().item()
        assert abs(score - expected) < 1e-3, (score, expected)

    # the result for an image doesn't depend on what else is in the batch
    for i, prompt in enumerate(prompts):
        single = {k: v[i:i + 1] for k, v in patches.items()}
        alone = generate_beam(model, single, num_beams=4, **{**kw, 'prompt_ids': [prompt]})
        assert alone == [out[i]]


def test_decode_objects_keep_identity_through_fsdp_input_casting():
    """ FSDP2 casts forward inputs by rebuilding known containers, state mutated inside layers must pass through
    by identity (plain classes). """
    from torch.distributed.utils import _apply_to_tensors
    from pixr.models import CrossKVSlot, CrossMemory, DecodeState
    from pixr.models.attention import CrossAttnMask
    mask = CrossAttnMask('flex', torch.ones(2, 4, dtype=torch.bool))
    memory = CrossMemory(torch.randn(2, 4, 8), mask, num_layers=2, batch_size=2)
    state = DecodeState([LayerKVCache()], torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, dtype=torch.long), memory)
    objs = (mask, memory, memory.slots[0], state, state.cache[0])
    cast = _apply_to_tensors(lambda t: t.to(torch.bfloat16) if t.is_floating_point() else t, objs)
    assert all(a is b for a, b in zip(objs, cast))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='varlen / flex cross-attention backends require CUDA')
@pytest.mark.parametrize('backend', ['varlen', 'flex'])
@pytest.mark.parametrize('kind', ['cruller', 'cruller_ettin', 'cruller_qwen3'])
def test_cross_attn_backends_w_hypothesis_groups(kind, backend):
    """ Packed (varlen) / flex cross-attention w/ several hypotheses per image match the dense sdpa backend. """
    from pixr.models.attention import has_flex, has_varlen
    if (backend == 'varlen' and not has_varlen) or (backend == 'flex' and not has_flex):
        pytest.skip(f'{backend} attention not available')
    ref = _gen_model(kind).cuda()
    cfg = replace(ref.cfg, text_decoder=replace(ref.cfg.text_decoder, cross_attn_backend=backend))
    model = create_model(cfg, vocab_size=100, pad_token_id=1).eval().cuda()
    model.load_state_dict(ref.state_dict())
    patches = {k: v.cuda() for k, v in _patches().items()}
    ids = torch.tensor([[5, 6], [9, 10], [7, 8], [11, 12]]).cuda()  # 2 prompts per image
    mask = torch.ones_like(ids, dtype=torch.bool)
    outs = []
    for m in (ref, model):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits, state = m.prefill(m.encode_image(patches), ids, mask)
            steps = [logits.float()]
            for parents, tokens in (([0, 0, 3, 2], [21, 22, 23, 24]), ([1, 0, 2, 2], [31, 32, 33, 34])):
                state.reorder(torch.tensor(parents).cuda())
                logits, state = m.step(torch.tensor(tokens).cuda()[:, None], state)
                steps.append(logits.float())
        outs.append(torch.stack(steps))
    torch.testing.assert_close(outs[1], outs[0], atol=5e-2, rtol=5e-2)


class _MarkovModel(torch.nn.Module):
    """ scripted next token distribution (depends on the last token only), for decoding algorithm tests """
    max_text_length = 64

    class _State:
        def reorder(self, parent_idx):
            return self

    def __init__(self, table):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('table', table)

    def encode_image(self, image_input):
        return None

    def prefill(self, memory, prompt_ids, prompt_mask):
        return self.table[prompt_ids[:, -1]], self._State()

    def step(self, token_ids, state):
        return self.table[token_ids[:, 0]], state


def test_beam_search_keeps_completed_hypotheses():
    """ A completed hypothesis must not be evicted by live prefixes w/ higher raw scores that finish worse. """
    from pixr.generation import generate_beam
    V, PAD, EOS, X, A, W, Y, Z, PROMPT = 16, 1, 2, 3, 4, 8, 5, 6, 15
    junk = list(range(9, 15))
    probs = torch.full((V, V), 1e-9)
    probs[PROMPT, X], probs[PROMPT, A] = 0.7, 0.3
    probs[A, EOS] = 0.67                      # [A, EOS]: log(.3) + log(.67) = -1.60, rank -0.80 at length 2
    probs[A, junk] = 0.33 / len(junk)
    probs[X, W] = 0.99                        # [X, W] = -0.37 stays ahead of the completed [A, EOS] in raw score
    probs[W, Y], probs[W, Z] = 0.51, 0.49     # both continuations (-1.06) beat [A, EOS]'s raw score
    spread = torch.linspace(1.3, 0.7, len(junk)) / len(junk)  # near uniform, no exact ties
    for t in (Y, Z, *junk):                   # ... and then only get worse, never finishing well
        probs[t, junk] = spread
    model = _MarkovModel(probs.log())
    kw = dict(prompt_ids=[[PROMPT]], stop_token_ids=[EOS], pad_token_id=PAD, max_new_tokens=6)
    out, scores = generate_beam(model, None, num_beams=2, return_scores=True, **kw)
    assert out == [[A, EOS]]
    assert abs(scores[0] - (math.log(0.3) + math.log(0.67))) < 1e-4
    # greedy (and one beam) follow the locally best token into the worse region
    assert generate_beam(model, None, num_beams=1, **kw) == generate_greedy(model, None, **kw)
    assert generate_greedy(model, None, **kw)[0][:2] == [X, W]


def test_one_beam_matches_greedy_on_stop_token_ties():
    """ An exact tie between a stop token and another token must resolve like greedy's argmax (first index). """
    from pixr.generation import generate_beam
    V, PAD, PROMPT, TOK, EOS = 100, 1, 5, 3, 90
    probs = torch.full((V, V), 1e-9)
    probs[PROMPT, TOK] = probs[PROMPT, EOS] = 0.5  # tie, greedy takes token 3 (lower index) and continues
    probs[TOK, EOS] = 1.
    model = _MarkovModel(probs.log())
    kw = dict(prompt_ids=[[PROMPT]], stop_token_ids=[EOS], pad_token_id=PAD, max_new_tokens=6)
    assert generate_greedy(model, None, **kw) == [[TOK, EOS]]
    assert generate_beam(model, None, num_beams=1, **kw) == [[TOK, EOS]]
    # and when the stop token has the lower index, both stop immediately
    probs2 = torch.full((V, V), 1e-9)
    probs2[PROMPT, 2] = probs2[PROMPT, TOK] = 0.5
    probs2[TOK, 2] = 1.
    model2 = _MarkovModel(probs2.log())
    kw2 = {**kw, 'stop_token_ids': [2]}
    assert generate_beam(model2, None, num_beams=1, **kw2) == generate_greedy(model2, None, **kw2) == [[2]]


def test_tiktoken_unused_ids_never_generated_or_decoded():
    from pixr.tokenizers import TokenizerCfg, create_tokenizer
    tok = create_tokenizer(TokenizerCfg(name='tiktoken:cl100k_base'))
    unused = tok.unused_token_ids
    # 16 ids w/o a token + the 5 cl100k_base special tokens (<|endoftext|>, fim prefix / middle / suffix, <|endofprompt|>)
    assert len(unused) == 21 and {100256, 100257, 100276} <= set(unused) and all(i < tok.vocab_size for i in unused)
    hello = tok.encode('hello world')
    assert tok.decode(hello[:1] + [100256] + hello[1:]) == 'hello world'  # ignored, no KeyError
    assert tok.decode(hello[:1] + [100257] + hello[1:]) == 'hello world'  # not decoded as '<|endoftext|>'
    assert tok.encode('a <|endoftext|> b') == tok.enc.encode_ordinary('a <|endoftext|> b')  # text, not the special
    # generation suppresses them even when they would win the argmax
    V, PROMPT, EOS = 8, 7, 2
    probs = torch.full((V, V), 1e-9)
    probs[PROMPT, 5], probs[PROMPT, 4] = 0.9, 0.1
    probs[4, EOS] = probs[5, EOS] = 1.
    model = _MarkovModel(probs.log())
    kw = dict(prompt_ids=[[PROMPT]], stop_token_ids=[EOS], pad_token_id=1, max_new_tokens=4)
    from pixr.generation import generate_beam
    assert generate_greedy(model, None, **kw) == [[5, EOS]]
    assert generate_greedy(model, None, bad_token_ids=[5], **kw) == [[4, EOS]]
    assert generate_beam(model, None, num_beams=2, bad_token_ids=[5], **kw) == [[4, EOS]]


def test_checkpoint_text_length_change(tmp_path):
    """ A pixr checkpoint loads into a model w/ a different (BART, learned position) text length. """
    from pixr.models import load_model_state
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = create_model(cfg, vocab_size=100, pad_token_id=1)
    state = model.state_dict()
    for max_length in (128, 32):
        resized = create_model(replace(cfg, text_decoder=replace(cfg.text_decoder, max_length=max_length)),
                               vocab_size=100, pad_token_id=1)
        load_model_state(resized, state)
        assert resized.text_decoder.embed_positions.weight.shape[0] == max_length + 2
        assert torch.equal(resized.text_decoder.embed_tokens.weight, state['text_decoder.embed_tokens.weight'])


def test_fritter_token_merge_rejects_patch_dropout():
    cfg = _tiny_cfg('fritter')
    kwargs = dict(cfg.image_encoder.model_kwargs, patch_drop_rate=0.5)
    with pytest.raises(ValueError, match='patch dropout'):
        create_model(replace(cfg, image_encoder=replace(cfg.image_encoder, model_kwargs=kwargs)),
                     vocab_size=100, pad_token_id=1)
    # w/o merging, patch dropout is fine
    create_model(replace(cfg, prefix_token_merge=1, image_encoder=replace(cfg.image_encoder, model_kwargs=kwargs)),
                 vocab_size=100, pad_token_id=1)
