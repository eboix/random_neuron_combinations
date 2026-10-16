"""Model adapters: which modules are the "neurons" (post-activation MLP hidden units) of each block, how to
turn a hooked output into a (B, H, W, D) spatial map, and the preprocessing the model expects.

ConvNeXt: encoder.stages.s.layers.b.act  -> (B, H, W, 4C) already spatial; stage-dependent focus crops.
ViT/DeiT: encoder.layer.i.intermediate.intermediate_act_fn -> (B, tokens, 4C); drop the CLS (and
          distillation) tokens and reshape the patch tokens to the 14x14 grid; 144 px focus crop (stride 16).
"""
import re

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
FOCUS_CONVNEXT = [64, 96, 144, 192]


def get_adapter(model):
    arch = model.config.model_type
    if arch == 'convnext':
        layers = []
        for name, mod in model.named_modules():
            m = re.fullmatch(r'convnext\.encoder\.stages\.(\d+)\.layers\.(\d+)\.act', name)
            if m:
                layers.append({'name': f'stage{m[1]}.block{m[2]}', 'slug': f'stage{m[1]}_block{m[2]}', 'stage': int(m[1]), 'block': int(m[2]),
                               'module': mod, 'module_name': name, 'focus': FOCUS_CONVNEXT[int(m[1])]})
        return {'arch': arch, 'layers': layers, 'to_spatial': lambda o: o,
                'preprocess': {'resize': 256, 'crop': 224, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}}
    if arch in ('vit', 'deit'):
        layers = []
        for name, mod in model.named_modules():
            m = re.fullmatch(rf'{arch}\.encoder\.layer\.(\d+)\.intermediate\.intermediate_act_fn', name)
            if m:
                layers.append({'name': f'layer{m[1]}', 'slug': f'layer{m[1]}', 'stage': 2, 'block': int(m[1]), 'module': mod, 'module_name': name, 'focus': 144})
        layers.sort(key=lambda l: l['block'])
        H = model.config.image_size // model.config.patch_size
        def to_spatial(o):
            B, T, D = o.shape
            return o[:, T - H * H:, :].reshape(B, H, H, D)      # drop CLS / distillation tokens (they come first)
        if arch == 'deit':   # DeiTImageProcessor: shortest edge 256, center crop 224, ImageNet mean/std
            pre = {'resize': 256, 'crop': 224, 'mean': IMAGENET_MEAN, 'std': IMAGENET_STD}
        else:                # ViTImageProcessor: direct resize to 224x224, mean/std 0.5
            pre = {'resize': [224, 224], 'crop': 224, 'mean': [0.5] * 3, 'std': [0.5] * 3}
        return {'arch': arch, 'layers': layers, 'to_spatial': to_spatial, 'preprocess': pre}
    raise ValueError(f'no adapter for architecture {arch}')


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
