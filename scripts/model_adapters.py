"""Model adapters: the hook sites of each block, how to turn a hooked output into a (B, H, W, D) spatial
map, and the preprocessing the model expects. Every site yields a list entry with name/slug/module/to_spatial.

ConvNeXt block (x -> dwconv7x7 -> LN -> pwconv1 -> GELU -> pwconv2 -> *gamma -> +x):
  mlp     post-GELU hidden units (4C)                   name  stageS.blockB        (the original "neurons")
  out     the block's write into the stream, gamma*pwconv2 (C)   name  stageS.blockB.out
  resid   the residual stream after the block (C)       name  stageS.blockB.resid
ViT/DeiT layer (x -> +attn(LN(x)) -> +mlp(LN(x))):
  mlp     post-GELU MLP hidden units (4D)               name  layerL
  attn_out  attention block's write (D)                 name  layerL.attn_out
  mlp_out   MLP block's write (D)                       name  layerL.mlp_out
  resid   residual stream after the layer (D)           name  layerL.resid
CLS/distillation tokens are dropped; the patch tokens form the 14x14 map (stride 16, 144 px focus crop).
"""
import re

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
FOCUS_CONVNEXT = [64, 96, 144, 192]
SITES_CONVNEXT = ['mlp', 'out', 'resid']
SITES_VIT = ['mlp', 'attn_out', 'mlp_out', 'resid']


def _entry(name, slug, site, stage, block, module, to_spatial, focus):
    return {'name': name, 'slug': slug, 'site': site, 'stage': stage, 'block': block, 'module': module, 'to_spatial': to_spatial, 'focus': focus}


def get_adapter(model, sites=None):
    arch = model.config.model_type
    mods = dict(model.named_modules())
    layers = []
    if arch == 'convnext':
        sites = sites or SITES_CONVNEXT
        for name, mod in model.named_modules():
            m = re.fullmatch(r'convnext\.encoder\.stages\.(\d+)\.layers\.(\d+)', name)
            if not m: continue
            s, b = int(m[1]), int(m[2]); base = f'stage{s}.block{b}'; bslug = f'stage{s}_block{b}'; focus = FOCUS_CONVNEXT[s]
            if 'mlp' in sites:
                layers.append(_entry(base, bslug, 'mlp', s, b, mods[name + '.act'], lambda o: o, focus))
            if 'out' in sites:
                layers.append(_entry(base + '.out', bslug + '_out', 'out', s, b, mods[name + '.pwconv2'],
                                     (lambda blk: (lambda o: o * blk.layer_scale_parameter))(mod), focus))
            if 'resid' in sites:
                layers.append(_entry(base + '.resid', bslug + '_resid', 'resid', s, b, mod,
                                     lambda o: (o[0] if isinstance(o, tuple) else o).permute(0, 2, 3, 1), focus))
        pre = {'resize': 256, 'crop': 224, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}
    elif arch in ('vit', 'deit'):
        sites = sites or SITES_VIT
        H = model.config.image_size // model.config.patch_size
        def tok(o):
            o = o[0] if isinstance(o, tuple) else o
            B, T, D = o.shape
            return o[:, T - H * H:, :].reshape(B, H, H, D)
        for name, mod in model.named_modules():
            m = re.fullmatch(rf'{arch}\.encoder\.layer\.(\d+)', name)
            if not m: continue
            b = int(m[1]); base = f'layer{b}'
            if 'mlp' in sites: layers.append(_entry(base, base, 'mlp', 2, b, mods[name + '.intermediate.intermediate_act_fn'], tok, 144))
            if 'attn_out' in sites: layers.append(_entry(base + '.attn_out', base + '_attn_out', 'attn_out', 2, b, mods[name + '.attention.output.dense'], tok, 144))
            if 'mlp_out' in sites: layers.append(_entry(base + '.mlp_out', base + '_mlp_out', 'mlp_out', 2, b, mods[name + '.output.dense'], tok, 144))
            if 'resid' in sites: layers.append(_entry(base + '.resid', base + '_resid', 'resid', 2, b, mod, tok, 144))
        layers.sort(key=lambda l: (l['block'], SITES_VIT.index(l['site'])))
        pre = ({'resize': 256, 'crop': 224, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD} if arch == 'deit'
               else {'resize': [224, 224], 'crop': 224, 'mean': [0.5] * 3, 'std': [0.5] * 3})
    else:
        raise ValueError(f'no adapter for architecture {arch}')
    return {'arch': arch, 'layers': layers, 'preprocess': pre, 'to_spatial': lambda o: o}


def find_layer(adapter, name):
    return next(l for l in adapter['layers'] if l['name'] == name)


def pil_input(im, pre):
    """Reproduce the model input (as a PIL image of size crop x crop) from a decoded PIL image."""
    from PIL import Image
    r = pre['resize']
    if isinstance(r, (list, tuple)):
        return im.resize((r[1], r[0]), Image.BILINEAR)
    w, h = im.size; s = r / min(w, h)
    im = im.resize((max(r, round(w * s)), max(r, round(h * s))), Image.BILINEAR)
    w, h = im.size; c = pre['crop']; l, t = (w - c) // 2, (h - c) // 2
    return im.crop((l, t, l + c, t + c))


def torch_transform(pre):
    import torchvision.transforms as T
    r = pre['resize']
    ops = [T.Resize(tuple(r), interpolation=T.InterpolationMode.BILINEAR)] if isinstance(r, (list, tuple)) else \
          [T.Resize(r, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(pre['crop'])]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(pre['mean'], pre['std'])])
