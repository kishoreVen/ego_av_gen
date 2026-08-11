"""
Object-centric PaliGemma patch similarity comparison.

For each pair of camera views, uses analytically-computed patch locations
(world-position → camera → patch grid) to extract and compare the specific
patch embeddings belonging to each object.

Install:
    uv pip install --python .venv/bin/python playwright matplotlib
    .venv/bin/playwright install chromium

Run from vlm_camera_test/:
    .venv/bin/python generate_comparison.py
"""

import asyncio, io, sys
from base64 import b64decode
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

SERVER     = 'http://localhost:8888'
OUT        = Path(__file__).parent / 'output'
OUT.mkdir(exist_ok=True)

FRUSTUM_W  = 3.8
FRUSTUM_H  = 2.8
DISP       = 512    # display size for render images (square)
CROP       = 72     # patch crop thumbnail size (px)
COLS       = 4      # objects per row in crop strip

CAMERAS = [
    ('center',   0.0,   0.0),
    ('x+0.6',   +0.6,   0.0),
    ('x+1.2',   +1.2,   0.0),
    ('x-1.0',   -1.0,   0.0),
    ('z+0.5',    0.0,  +0.5),
    ('z-0.5',    0.0,  -0.5),
    ('diag',    +0.8,  +0.4),
]

SIM_CMAP = LinearSegmentedColormap.from_list('sim',
    [(0.0, '#2244cc'), (0.5, '#eeeeee'), (1.0, '#cc2222')])

# ── Browser ─────────────────────────────────────────────────────────────────

async def wait_ready(page):
    print('  waiting for model…')
    await page.wait_for_selector('#compute-btn:not([disabled])', timeout=180_000)
    await page.wait_for_timeout(6_000)   # GLB load time
    print('  ready.')

async def capture_view(page, cam_x: float, cam_z: float) -> dict:
    await page.evaluate(f"""(() => {{
        const ids = ['cam-x','cam-z'], vals = [{cam_x},{cam_z}];
        ids.forEach((id,i) => {{
            const el = document.getElementById(id);
            el.value = vals[i];
            el.dispatchEvent(new Event('input'));
        }});
    }})()""")
    await page.wait_for_timeout(300)

    render_b64 = await page.evaluate(
        "document.getElementById('three-canvas').toDataURL('image/png').split(',')[1]"
    )
    obj_reg = await page.evaluate("""
        typeof window._objRegistry !== 'undefined'
            ? window._objRegistry.map(o => ({name:o.name, x:o.worldX, z:o.worldZ, color:o.color}))
            : []
    """)

    resp = requests.post(f'{SERVER}/embed', json={'image': render_b64}, timeout=120)
    resp.raise_for_status()
    r = resp.json()
    r['render_b64'] = render_b64
    r['obj_registry'] = obj_reg
    r['cam_x'] = cam_x
    r['cam_z'] = cam_z
    return r

# ── Patch math ───────────────────────────────────────────────────────────────

def world_to_patch(ox, oz, cam_x, cam_z, g):
    """Return (col, row, linear_idx, in_bounds) for an object in world space."""
    u = (ox - cam_x) / FRUSTUM_W + 0.5
    v = (oz - cam_z) / FRUSTUM_H + 0.5
    col = int(np.clip(u * g, 0, g - 1))
    row = int(np.clip(v * g, 0, g - 1))
    return col, row, row * g + col, (0.0 <= u < 1.0 and 0.0 <= v < 1.0)

def crop_patch(img: Image.Image, col: int, row: int, g: int, size: int) -> Image.Image:
    """Crop and upscale a single patch region from a square DISP×DISP image."""
    cell = img.width / g
    x0, y0 = int(col * cell), int(row * cell)
    x1, y1 = int((col+1) * cell), int((row+1) * cell)
    return img.crop((x0, y0, x1, y1)).resize((size, size), Image.LANCZOS)

# ── Per-object data ──────────────────────────────────────────────────────────

def compute_objects(va: dict, vb: dict) -> list:
    """
    For every object visible in both views, compute:
      - patch (col,row) in view A and B
      - patch embedding cosine similarity (SigLIP space)
      - patch image crops
    """
    g  = va['grid_size']
    ea = np.array(va['patch_embeddings_norm'], dtype=np.float32)
    eb = np.array(vb['patch_embeddings_norm'], dtype=np.float32)

    img_a = Image.open(io.BytesIO(b64decode(va['render_b64']))).convert('RGB').resize((DISP, DISP), Image.LANCZOS)
    img_b = Image.open(io.BytesIO(b64decode(vb['render_b64']))).convert('RGB').resize((DISP, DISP), Image.LANCZOS)

    objs = []
    for obj in va['obj_registry']:
        col_a, row_a, idx_a, ok_a = world_to_patch(obj['x'], obj['z'], va['cam_x'], va['cam_z'], g)
        col_b, row_b, idx_b, ok_b = world_to_patch(obj['x'], obj['z'], vb['cam_x'], vb['cam_z'], g)

        sim = float(np.dot(ea[idx_a], eb[idx_b])) if ok_a and ok_b else None

        objs.append(dict(
            name   = obj['name'],
            color  = obj['color'],
            col_a=col_a, row_a=row_a, idx_a=idx_a, ok_a=ok_a,
            col_b=col_b, row_b=row_b, idx_b=idx_b, ok_b=ok_b,
            sim    = sim,
            crop_a = crop_patch(img_a, col_a, row_a, g, CROP) if ok_a else None,
            crop_b = crop_patch(img_b, col_b, row_b, g, CROP) if ok_b else None,
            img_a  = img_a,
            img_b  = img_b,
        ))

    return sorted([o for o in objs if o['sim'] is not None], key=lambda o: -o['sim'])

# ── Annotated render ─────────────────────────────────────────────────────────

def sim_to_hex(s):
    t = (float(np.clip((s+1)/2, 0, 1)))
    r = int(t * 200 + 55)
    b = int((1-t) * 200 + 55)
    g = int(min(t, 1-t) * 2 * 140 + 20)
    return f'#{r:02x}{g:02x}{b:02x}'

def annotate_render(img: Image.Image, objs: list, view: str, g: int) -> Image.Image:
    """Draw patch boxes coloured by similarity on the render."""
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    cell = DISP / g

    for o in objs:
        col = o[f'col_{view}']
        row = o[f'row_{view}']
        ok  = o[f'ok_{view}']
        if not ok:
            continue
        sim   = o['sim']
        color = sim_to_hex(sim) if sim is not None else '#888888'
        x0, y0 = col * cell, row * cell
        x1, y1 = x0 + cell, y0 + cell

        # Box outline (thicker = more visible)
        draw.rectangle([x0+1, y0+1, x1-1, y1-1], outline=color, width=3)

        # Sim label inside box
        label = f'{sim:.2f}' if sim is not None else '?'
        draw.rectangle([x0+2, y0+2, x0+28, y0+13], fill=(0,0,0,180))
        draw.text((x0+3, y0+2), label, fill=color)

    return out

# ── Crop comparison strip (PIL) ───────────────────────────────────────────────

def make_crop_strip(objs: list) -> Image.Image:
    """
    Build a PIL image grid: for each object one cell showing
    [crop_A | divider | crop_B] with object name and cosine similarity.
    """
    n     = len(objs)
    rows  = (n + COLS - 1) // COLS
    PAD   = 8
    LABEL = 22
    cell_w = CROP * 2 + PAD * 3   # A | gap | B
    cell_h = CROP + LABEL + PAD

    strip = Image.new('RGB', (COLS * cell_w + PAD, rows * cell_h + PAD), (14, 14, 22))
    draw  = ImageDraw.Draw(strip)

    for i, o in enumerate(objs):
        r, c = divmod(i, COLS)
        x = c * cell_w + PAD
        y = r * cell_h + PAD

        if o['crop_a']:
            strip.paste(o['crop_a'], (x, y))
        else:
            draw.rectangle([x, y, x+CROP, y+CROP], fill=(40,40,60))
            draw.text((x+4, y+CROP//2), 'off\nscreen', fill='#666')

        if o['crop_b']:
            strip.paste(o['crop_b'], (x + CROP + PAD, y))
        else:
            draw.rectangle([x+CROP+PAD, y, x+CROP*2+PAD, y+CROP], fill=(40,40,60))
            draw.text((x+CROP+PAD+4, y+CROP//2), 'off\nscreen', fill='#666')

        # Thin divider between crops
        draw.line([(x+CROP+PAD//2, y), (x+CROP+PAD//2, y+CROP)], fill='#333', width=1)

        # Similarity colour bar under the pair
        sim  = o['sim']
        t    = (sim + 1) / 2
        r_   = int(t * 200 + 55)
        b_   = int((1-t) * 200 + 55)
        g_   = int(min(t, 1-t) * 2 * 140 + 20)
        bar_w = int((CROP * 2 + PAD) * t)
        draw.rectangle([x, y+CROP+2, x + bar_w, y+CROP+6], fill=(r_, g_, b_))
        draw.rectangle([x, y+CROP+2, x + CROP*2+PAD, y+CROP+6], outline='#333')

        # Label
        short = o['name'][:9]
        label = f'{short}  cos={sim:+.3f}'
        draw.text((x, y + CROP + 8), label, fill=o['color'])

    return strip

# ── Main comparison figure ───────────────────────────────────────────────────

def make_comparison(va: dict, vb: dict, label_a: str, label_b: str, out_path: Path):
    g    = va['grid_size']
    objs = compute_objects(va, vb)
    if not objs:
        print(f'  no objects in view — skipping {out_path.name}')
        return

    ann_a = annotate_render(Image.open(io.BytesIO(b64decode(va['render_b64']))).convert('RGB').resize((DISP,DISP),Image.LANCZOS), objs, 'a', g)
    ann_b = annotate_render(Image.open(io.BytesIO(b64decode(vb['render_b64']))).convert('RGB').resize((DISP,DISP),Image.LANCZOS), objs, 'b', g)

    strip = make_crop_strip(objs)

    # ── Compose figure ─────────────────────────────────────────────────────
    fig  = plt.figure(figsize=(22, 15), facecolor='#0e0e18')
    gs   = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.0, 0.9],
                             hspace=0.06, wspace=0.04,
                             left=0.01, right=0.99,
                             top=0.93, bottom=0.01)

    def show(pos, arr, title):
        ax = fig.add_subplot(pos)
        ax.imshow(np.array(arr))
        ax.set_title(title, color='#dddddd', fontsize=11, pad=5, loc='left')
        ax.axis('off')
        return ax

    # Row 0: two renders + bar chart
    show(gs[0, 0], ann_a,
         f'View A — {label_a}\n'
         f'Box colour = per-object cosine sim vs View B')
    show(gs[0, 1], ann_b,
         f'View B — {label_b}\n'
         f'Same colour scale')

    ax_bar = fig.add_subplot(gs[0, 2])
    ax_bar.set_facecolor('#080810')
    names  = [o['name'][:11]  for o in objs]
    sims   = [o['sim']         for o in objs]
    colors = [sim_to_hex(s)    for s in sims]
    ys     = range(len(names))
    bars   = ax_bar.barh(list(ys), sims, color=colors, height=0.6, alpha=0.9)
    ax_bar.set_yticks(list(ys))
    ax_bar.set_yticklabels(names, color='#cccccc', fontsize=9)
    ax_bar.set_xlim(-0.1, 1.1)
    ax_bar.axvline(0, color='#444', lw=0.8)
    ax_bar.axvline(1, color='#333', lw=0.5, ls='--')
    for bar, val in zip(bars, sims):
        ax_bar.text(min(val + 0.02, 1.05),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:+.3f}', va='center', ha='left',
                    color='white', fontsize=9, fontweight='bold')
    ax_bar.set_xlabel('object-patch cosine similarity  (A patch ↔ B patch, same object)',
                      color='#888', fontsize=9)
    ax_bar.set_title('Per-object patch similarity\n'
                     'same object, different image position',
                     color='#dddddd', fontsize=10, pad=5)
    ax_bar.tick_params(colors='#888')
    for sp in ['top', 'right']:
        ax_bar.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax_bar.spines[sp].set_color('#333')

    # Row 1: crop strip spanning all columns
    ax_strip = fig.add_subplot(gs[1, :])
    ax_strip.imshow(np.array(strip))
    ax_strip.set_title(
        f'Object patch crops  |  left=View A  right=View B  |  '
        f'colour bar = similarity  |  sorted high→low',
        color='#aaaaaa', fontsize=9, pad=4, loc='left')
    ax_strip.axis('off')

    # Camera shift info
    dx = vb['cam_x'] - va['cam_x']
    dz = vb['cam_z'] - va['cam_z']
    sims_arr = np.array(sims)
    fig.suptitle(
        f'PaliGemma-3B  SigLIP  |  '
        f'camera shift  Δx={dx:+.1f}  Δz={dz:+.1f}  |  '
        f'{g}×{g} patches  |  '
        f'mean obj-patch sim = {sims_arr.mean():.3f}  '
        f'(min={sims_arr.min():.3f}  max={sims_arr.max():.3f})',
        color='white', fontsize=12, y=0.96
    )

    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='#0e0e18')
    plt.close(fig)
    print(f'  → {out_path.name}')

# ── Summary grid ─────────────────────────────────────────────────────────────

def make_summary(results: dict, labels: dict):
    names = list(results.keys())
    N     = len(names)

    # Matrix: mean per-object patch similarity (only objects visible in both views)
    mat = np.full((N, N), np.nan)
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            va, vb = results[na], results[nb]
            g  = va['grid_size']
            ea = np.array(va['patch_embeddings_norm'], dtype=np.float32)
            eb = np.array(vb['patch_embeddings_norm'], dtype=np.float32)
            sims = []
            for obj in va['obj_registry']:
                _, _, idx_a, ok_a = world_to_patch(obj['x'], obj['z'], va['cam_x'], va['cam_z'], g)
                _, _, idx_b, ok_b = world_to_patch(obj['x'], obj['z'], vb['cam_x'], vb['cam_z'], g)
                if ok_a and ok_b:
                    sims.append(float(np.dot(ea[idx_a], eb[idx_b])))
            mat[i, j] = np.mean(sims) if sims else np.nan

    fig, ax = plt.subplots(figsize=(N * 1.5 + 2, N * 1.3 + 1.5), facecolor='#0e0e18')
    ax.set_facecolor('#0e0e18')
    im = ax.imshow(mat, cmap=SIM_CMAP, vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([labels[n] for n in names], rotation=40,
                       ha='right', color='#cccccc', fontsize=9)
    ax.set_yticklabels([labels[n] for n in names], color='#cccccc', fontsize=9)
    for i in range(N):
        for j in range(N):
            v = mat[i, j]
            c = 'black' if 0.3 < v < 0.8 else 'white'
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    color=c, fontsize=9, fontweight='bold')
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('mean per-object patch cosine similarity', color='#aaa')
    cb.ax.yaxis.set_tick_params(color='#aaa')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')
    ax.set_title(
        'Pairwise object-patch similarity — PaliGemma SigLIP\n'
        '(each cell = mean cosine sim between object patches across both views)',
        color='white', fontsize=12, pad=10
    )
    out = OUT / 'summary_grid.png'
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='#0e0e18')
    plt.close(fig)
    print(f'  → {out.name}')

# ── Entry ────────────────────────────────────────────────────────────────────

async def run():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit('Run:  uv pip install --python .venv/bin/python playwright\n'
                 '      .venv/bin/playwright install chromium')

    print('Launching browser…')
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox',
                  '--disable-dev-shm-usage', '--ignore-gpu-blocklist',
                  '--use-gl=angle', '--use-angle=swiftshader']
        )
        page = await browser.new_page(viewport={'width': 1400, 'height': 900})
        await page.goto(SERVER)
        await wait_ready(page)

        results, labels = {}, {}
        for name, cam_x, cam_z in CAMERAS:
            print(f'\nCapturing  {name}  (x={cam_x:+.1f}  z={cam_z:+.1f})…')
            results[name] = await capture_view(page, cam_x, cam_z)
            labels[name]  = f'{name} (x={cam_x:+.1f}, z={cam_z:+.1f})'
            n_obj = len(results[name]['obj_registry'])
            print(f'  {results[name]["grid_size"]}×{results[name]["grid_size"]} patches  {n_obj} objects')

    baseline = CAMERAS[0][0]
    print(f'\nGenerating comparisons (baseline = {baseline})…')
    for name, cam_x, cam_z in CAMERAS[1:]:
        make_comparison(
            results[baseline], results[name],
            labels[baseline], labels[name],
            OUT / f'compare_{baseline}_vs_{name}.png'
        )

    print('\nGenerating summary grid…')
    make_summary(results, labels)
    print(f'\nDone → {OUT}/')

if __name__ == '__main__':
    asyncio.run(run())
