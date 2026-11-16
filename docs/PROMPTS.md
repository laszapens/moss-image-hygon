# MOSS-Image prompt recipes

Tested on Hygon BW 64GB at 1024×1024, 20 steps, guidance 3.0, seed varies.

MOSS-Image accepts Danbooru-style tag sequences, natural language, and JSON
layout descriptions. For anime subjects, tags plus a plain-language scene
description works best.

## Base tags

Include these in essentially every prompt:

```
masterpiece, best quality, highly detailed, absurdres, newest
```

## Negative prompt

Only takes effect when `guidance_scale > 1`. At guidance 1.0 there is no CFG and
the negative branch is never evaluated.

```
nsfw, worst quality, low quality, bad anatomy, bad hands, extra fingers, watermark, signature, jpeg artifacts
```

## Settings

| Goal | Steps | Guidance | Time at 1024² |
|---|---|---|---|
| Fast drafts | 20 | 1.0 | ~9.6 s |
| Normal use | 20 | 3.0 | ~19 s |
| High detail | 30 | 3.0–4.0 | ~33 s |
| Large output | 30 | 3.0 | ~38 s at 1328² |

20 and 50 steps are visually indistinguishable for anime subjects on this
hardware, so the upstream default of 50 is usually wasted time. Guidance 1.0 is
roughly twice as fast but visibly flatter in color.

---

## Recipes

### 1. Japanese atmosphere

Consistently the strongest results; layered light and shadow.

```
masterpiece, best quality, highly detailed, anime illustration of a girl in a
white cheongsam with golden embroidery, long black hair, red eyes, standing in
a bamboo forest, dappled sunlight through leaves, god rays, intricate hair
ornament with tassel, cel shading, detailed eyes, cinematic lighting
```

### 2. Cyberpunk neon

Plays to the model's saturated color handling.

```
masterpiece, best quality, highly detailed, cyberpunk anime girl with neon blue
twin tails, holographic jacket, standing in a rainy neon-lit alley at night,
reflections on wet ground, glowing signs, bokeh, chromatic aberration
```

### 3. Cherry blossom portrait

The most reliable moe formula; rarely fails.

```
masterpiece, best quality, highly detailed, anime illustration of a young girl
with long silver hair and blue eyes, white sailor uniform, under cherry
blossoms, petals falling, soft sunlight, detailed eyes, cel shading, clean
lineart, vibrant colors
```

### 4. Combat action

Tests structural coherence under motion.

```
masterpiece, best quality, highly detailed, dynamic action shot, anime girl
with crimson hair wielding a glowing katana, mid-slash, sparks and debris
flying, dramatic perspective, motion blur, intense expression, detailed armor
```

### 5. Slice of life

Soft lighting; reveals color transitions.

```
masterpiece, best quality, highly detailed, cozy anime scene, girl with brown
bob hair reading a book by a window, afternoon sunlight, dust motes in the air,
warm color palette, soft shading, detailed interior, slice of life
```

### 6. Chinese classical style

Uses the model's Chinese-language corpus advantage. Mixing Chinese description
with English quality tags works better than pure Chinese.

```
masterpiece, best quality, 古风插画，一位身着红色广袖流仙裙的少女，长发及腰，
站在雪中红梅树下，水墨风格的远山，飘落的雪花，精致发簪，工笔画质感，细腻笔触
```

### 7. Rendered Chinese text

MOSS-Image renders text correctly, including Chinese handwriting — a genuine
differentiator. State the language and exact content.

```
anime illustration, masterpiece, best quality, a cheerful girl with pink twin
tails holding a wooden sign that reads "欢迎光临" in bold Chinese characters,
colorful festival background with lanterns and sparkles, cel shading, clean
lineart
```

### 8. Small text on an object

Harder text test; a book title rather than a sign.

```
anime illustration, masterpiece, best quality, a bookstore scene, a girl
holding a book titled "雪国", warm lighting, detailed bookshelves, soft shading
```

### 9. Handwritten note (upstream reference)

The upstream model card prompt. Renders three lines of handwritten Chinese on a
sticky note, character-perfect.

```
A wide view of a refrigerator beside a kitchen counter with a fruit bowl.
A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"
```

---

## Notes

- Text rendering quality degrades with distance and small font size. For
  reliable results, make the text large in frame and name the surface explicitly
  ("a sign", "a book cover", "a sticky note").
- `uncond_skip_layers=[22]` (the default) is skip-layer guidance on the
  unconditional branch. It is a compute saving, not a prompt feature. Don't set it
  to `None` expecting better quality; the difference is minor and the cost is real.
- Mixing Chinese scene description with English quality tags performs better than
  either language alone.
- For a different composition at the same prompt, change the seed. Seeds are
  reported back in the UI so any result is reproducible.
