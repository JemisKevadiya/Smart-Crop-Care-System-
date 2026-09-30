"""Generate assets/hero_leaves.jpg, the leafy banner background used by app.py.

Procedurally drawn (no third-party images), so there are no licensing concerns.

    python scripts/make_hero_image.py
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "hero_leaves.jpg"
W, H = 1800, 600
SCALE = 2  # draw at 2x and downsample for smooth edges


def leaf_points(cx, cy, length, width, angle, steps=24):
    """Outline of a pointed leaf (two arcs meeting at tip and base)."""
    pts = []
    for side in (1, -1):
        rng = range(steps + 1) if side == 1 else range(steps, -1, -1)
        for i in rng:
            t = i / steps                                  # 0 = base, 1 = tip
            x = (t - 0.5) * length
            y = side * width / 2 * math.sin(math.pi * t) ** 0.9 * (1 - 0.25 * t)
            pts.append((x, y))
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in pts]


def draw_leaf(draw, cx, cy, length, angle, shade):
    width = length * random.uniform(0.38, 0.5)
    r, g, b = shade
    draw.polygon(leaf_points(cx, cy, length, width, angle), fill=(r, g, b))
    # lighter midrib and side veins
    vein = (min(255, r + 70), min(255, g + 90), min(255, b + 60))
    ca, sa = math.cos(angle), math.sin(angle)
    base = (cx - length / 2 * ca, cy - length / 2 * sa)
    tip = (cx + length / 2 * ca, cy + length / 2 * sa)
    draw.line([base, tip], fill=vein, width=max(2, int(length / 60)))
    for k in range(1, 6):
        t = k / 6.5
        px, py = base[0] + (tip[0] - base[0]) * t, base[1] + (tip[1] - base[1]) * t
        for side in (1, -1):
            a = angle + side * 0.9
            vl = width * 0.42 * (1 - 0.4 * t)
            draw.line([(px, py), (px + vl * math.cos(a), py + vl * math.sin(a))],
                      fill=vein, width=max(1, int(length / 110)))


def layer(count, size_range, shades, blur):
    img = Image.new("RGBA", (W * SCALE, H * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for _ in range(count):
        draw_leaf(draw,
                  random.uniform(-60, W + 60) * SCALE, random.uniform(-60, H + 60) * SCALE,
                  random.uniform(*size_range) * SCALE, random.uniform(0, 2 * math.pi),
                  random.choice(shades))
    img = img.resize((W, H), Image.LANCZOS)
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


def main():
    random.seed(7)
    canvas = Image.new("RGBA", (W, H), (6, 28, 14, 255))
    # back to front: darker, blurrier leaves behind; brighter, sharper in front
    canvas.alpha_composite(layer(260, (70, 130), [(12, 52, 24), (16, 62, 30), (10, 45, 22)], 3))
    canvas.alpha_composite(layer(200, (90, 160), [(22, 88, 40), (28, 100, 46), (18, 76, 36)], 1.2))
    canvas.alpha_composite(layer(120, (110, 190), [(36, 122, 56), (46, 138, 62), (30, 110, 50)], 0))
    OUT.parent.mkdir(exist_ok=True)
    canvas.convert("RGB").save(OUT, quality=82, optimize=True, progressive=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
