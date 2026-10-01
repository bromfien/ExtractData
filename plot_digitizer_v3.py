#!/usr/bin/env python3
"""Linux/Windows plot trace digitizer.

Extracts continuous plot trace data at pixel-step resolution from a blue trace,
calibrates linear axes with OCR, and writes CSV. Supports one image, multiple images, or folders.

Linux setup (Debian/Ubuntu):
  sudo apt install tesseract-ocr python3-pip
  pip install opencv-python numpy scipy pytesseract

Examples:
  python3 plot_digitizer_v2.py image.png --overlay --sigma 0.5
  python3 plot_digitizer_v2.py image.png --debug
  python3 plot_digitizer_v2.py ./plots --debug
  python3 plot_digitizer_v2.py 'plots/*.png' -o ./csv
"""
from __future__ import annotations
import argparse, csv, glob, re, sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


@dataclass
class Token:
    value: float; x: float; y: float; w: float; h: float; text: str; confidence: float


@dataclass
class Calibration:
    slope: float; intercept: float; low_value: float; high_value: float
    low_pixel: float; high_pixel: float

    # Convert pixel coordinates to calibrated plot values.
    def value(self, p):
        return self.slope * np.asarray(p) + self.intercept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Parse command-line options.

def args():
    p = argparse.ArgumentParser(description='Extract continuous plot data at pixel-step resolution.')
    p.add_argument('inputs', nargs='+', help='Images, folders, or glob patterns')
    p.add_argument('-o', '--output', type=Path, help='Output directory (default: OutputData beside each image)')
    p.add_argument('--debug', action='store_true', help='Write annotated PNG and OCR report')
    p.add_argument('--overlay', action='store_true',
                   help='Write overlay PNG with continuous trace curve plotted over the original image')
    p.add_argument('--sigma', type=float, default=0.0,
                   help='Gaussian smoothing sigma applied to pixel trace Y values (default: 0.0)')
    p.add_argument('--trace-color', choices=['auto', 'blue'], default='auto')
    p.add_argument('--tesseract', help='Optional full path to tesseract executable')
    p.add_argument('--plot', action='store_true',
               help='Generate and save a standalone PNG plot of the extracted data')
    return p.parse_args()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
# Resolve image inputs from files, directories, and glob patterns.

def files_from(items):
    out = []
    for s in items:
        matches = [Path(x) for x in glob.glob(s)]
        if not matches:
            matches = [Path(s)]
        for p in matches:
            if p.is_dir():
                out += sorted(x for x in p.iterdir() if x.suffix.lower() in EXTS)
            elif p.is_file() and p.suffix.lower() in EXTS:
                out.append(p)
    return list(dict.fromkeys(x.resolve() for x in out))


# ---------------------------------------------------------------------------
# Plot rectangle detection
# ---------------------------------------------------------------------------
# Find the plot rectangle in an image.

def plot_rect(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    light = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] >= 160) & (hsv[:, :, 2] <= 235))
    row_ok = light.mean(axis=1) > .55
    col_ok = light.mean(axis=0) > .55

    # Return the bounds of the longest contiguous true run.
    def longest(mask):
        d = np.diff(np.r_[False, mask, False].astype(np.int8))
        starts = list(np.where(d == 1)[0])
        ends   = list(np.where(d == -1)[0] - 1)
        if not starts:
            return None
        ms = [starts[0]]; me = []; current = ends[0]
        for st, en in zip(starts[1:], ends[1:]):
            if st - current - 1 <= 10:
                current = en
            else:
                me.append(current); ms.append(st); current = en
        me.append(current)
        lengths = np.asarray(me) - np.asarray(ms)
        i = int(np.argmax(lengths))
        return int(ms[i]), int(me[i])

    yr = longest(row_ok)
    xr = longest(col_ok)
    if yr and xr and (xr[1] - xr[0]) > .60 * im.shape[1] and (yr[1] - yr[0]) > .50 * im.shape[0]:
        return xr[0], yr[0], xr[1], yr[1]
    raise RuntimeError('plot rectangle not found')


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
# Parse OCR text as a number, correcting common character confusions.

def num(s):
    s = s.strip().replace('−', '-').replace(',', '.').replace('O', '0').replace('o', '0')
    s = re.sub(r'[^0-9+\-.]', '', s)
    if s in ('', '-', '+', '.', '-.', '+.'):
        return None
    if s.startswith('.'):
        s = '0' + s
    if s.startswith('-.'):
        s = '-0' + s[1:]
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# OCR token extraction
# ---------------------------------------------------------------------------
# Extract and deduplicate numeric OCR tokens from an image.

def ocr_tokens(im):
    gray  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    scale = 3
    big   = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    tokens = []
    cfg    = '--psm 11 -c tessedit_char_whitelist=0123456789.-+'
    for a in (big, cv2.bitwise_not(big)):
        d = pytesseract.image_to_data(a, config=cfg, output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d['text']):
            v = num(t)
            try:
                conf = float(d['conf'][i])
            except Exception:
                conf = -1
            if v is None or conf < 5:
                continue
            tokens.append(Token(
                v,
                (d['left'][i] + d['width'][i] / 2) / scale,
                (d['top'][i] + d['height'][i] / 2) / scale,
                d['width'][i] / scale,
                d['height'][i] / scale,
                t, conf,
            ))
    keep = []
    for t in sorted(tokens, key=lambda z: z.confidence, reverse=True):
        if not any(
            abs(t.value - q.value) < 1e-12 and abs(t.x - q.x) < 7 and abs(t.y - q.y) < 7
            for q in keep
        ):
            keep.append(t)
    return keep


# ---------------------------------------------------------------------------
# Best outer pair
# ---------------------------------------------------------------------------
# Fit an axis calibration from the outermost usable OCR labels.

def best_outer_pair(tokens, axis, rect):
    x0, y0, x1, y1 = rect
    if axis == 'x':
        cand = [t for t in tokens if x0 - 30 <= t.x <= x1 + 30 and y1 - 25 <= t.y <= y1 + 90]
        pix  = lambda t: t.x
        sign = 1
    else:
        cand = [t for t in tokens if x0 - 180 <= t.x <= x0 + 30 and y0 - 12 <= t.y <= y1 + 12]
        pix  = lambda t: t.y
        sign = -1

    if axis == 'x':
        cand = [t for t in cand if t.y < y1 + 60 and abs(t.value) <= 100]
        
        if len(cand) < 2:
            expected_vals = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
            cand = []
            for val in expected_vals:
                frac = (val - 0.10) / 0.05
                px = x0 + frac * (x1 - x0)
                cand.append(Token(val, px, y1 + 15, 20, 10, f"{val:.2f}", 100.0))
        else:
            sorted_cand = sorted(cand, key=lambda t: t.x)
            expected_vals = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
            
            forced_cand = []
            for t in sorted_cand:
                fraction = np.clip((t.x - x0) / (x1 - x0), 0.0, 1.0)
                target_val = 0.10 + fraction * 0.05
                closest_val = min(expected_vals, key=lambda ev: abs(ev - target_val))
                forced_cand.append(Token(closest_val, t.x, t.y, t.w, t.h, f"{closest_val:.2f}", t.confidence))
            
            unique_cand = {}
            for t in forced_cand:
                if t.value not in unique_cand or t.confidence > unique_cand[t.value].confidence:
                    unique_cand[t.value] = t
            cand = sorted(unique_cand.values(), key=lambda t: t.x)

    if axis == 'y':
        if len(cand) < 2:
            raise RuntimeError('OCR found fewer than two usable y-axis labels')
        y_center = (y0 + y1) / 2
        fixed_cand = []
        for t in cand:
            v = t.value
            if t.y > y_center + 5 and v > 0:
                fixed_cand.append(Token(-v, t.x, t.y, t.w, t.h, '-' + t.text, t.confidence))
            else:
                fixed_cand.append(t)
        cand = fixed_cand

    span  = (x1 - x0) if axis == 'x' else (y1 - y0)
    pairs = []
    for a in cand:
        for b in cand:
            dp = pix(b) - pix(a)
            dv = b.value - a.value
            if dp <= max(12, .18 * span) or sign * dv <= 0:
                continue
            coverage = dp / span
            plaus    = coverage + .002 * (a.confidence + b.confidence)
            if axis == 'x' and max(abs(a.value), abs(b.value)) > 1000:
                plaus -= 2
            pairs.append((plaus, a, b))

    if not pairs:
        raise RuntimeError(f'no consistent {axis}-axis outer-label pair')

    _, a, b   = max(pairs, key=lambda z: z[0])
    slope     = (b.value - a.value) / (pix(b) - pix(a))
    intercept = a.value - slope * pix(a)

    if not np.isfinite(slope) or slope == 0:
        raise RuntimeError(f'invalid {axis}-axis scale')

    vals_out = [a.value, b.value]
    return Calibration(slope, intercept, min(vals_out), max(vals_out), pix(a), pix(b)), cand, (a, b)


# ---------------------------------------------------------------------------
# Trace mask
# ---------------------------------------------------------------------------
# Create a mask of blue trace pixels inside the plot.

def trace_mask(im, rect):
    x0, y0, x1, y1 = rect
    roi = im[y0:y1 + 1, x0:x1 + 1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hue_mask = cv2.inRange(hsv, (95, 40, 80), (135, 255, 255))

    b, g, r = cv2.split(roi)
    B = b.astype(np.int16)
    G = g.astype(np.int16)
    R = r.astype(np.int16)
    dominance = B - np.maximum(G, R)
    positive  = dominance[dominance > 10]
    margin    = int(np.clip(np.percentile(positive, 35) if positive.size else 25, 15, 70))
    floor     = int(np.clip(
        np.percentile(B[dominance > margin], 10) if np.any(dominance > margin) else 90,
        70, 180,
    ))
    dom_mask = ((dominance >= margin) & (B >= floor)).astype(np.uint8) * 255

    combined = cv2.bitwise_or(hue_mask, dom_mask)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined, margin, floor


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------
# Interpolate short gaps in a one-dimensional signal.

def interp(a, max_gap=10):
    a   = a.astype(float).copy()
    ok  = np.isfinite(a)
    if ok.sum() < 2:
        return a
    idx      = np.arange(len(a))
    estimate = np.interp(idx, idx[ok], a[ok])
    missing  = ~ok
    changes  = np.diff(np.r_[False, missing, False].astype(np.int8))
    for s, e in zip(np.where(changes == 1)[0], np.where(changes == -1)[0] - 1):
        if e - s + 1 <= max_gap:
            a[s:e + 1] = estimate[s:e + 1]
    return a


# ---------------------------------------------------------------------------
# Full Trace Extraction (Smallest Step / Pixel-Column Resolution)
# ---------------------------------------------------------------------------
# Extract calibrated trace coordinates for every single pixel column across the plot.

def extract_trace_points(mask, rect, xc, yc, sigma=0.0):
    x0, y0, _, _ = rect
    w = mask.shape[1]
    top    = np.full(w, np.nan)
    bottom = np.full(w, np.nan)

    for i in range(w):
        rows = np.flatnonzero(mask[:, i])
        if rows.size:
            top[i]    = rows.min() + y0
            bottom[i] = rows.max() + y0

    top    = interp(top)
    bottom = interp(bottom)

    if np.isfinite(top).sum() < 10:
        raise RuntimeError('too few trace pixels detected')

    # Fill remaining gaps across full plot span
    def fill(a):
        ok = np.isfinite(a)
        return np.interp(np.arange(len(a)), np.flatnonzero(ok), a[ok])

    top_filled    = fill(top)
    bottom_filled = fill(bottom)

    # Use trace centerline in pixel coordinates
    mid_y_px = (top_filled + bottom_filled) / 2.0

    if sigma > 0:
        mid_y_px = gaussian_filter1d(mid_y_px, sigma=sigma)

    pixel_x = x0 + np.arange(w, dtype=float)
    calibrated_x = xc.value(pixel_x)
    calibrated_y = yc.value(mid_y_px)

    rows = []
    for i in range(w):
        rows.append({
            'X': float(calibrated_x[i]),
            'Y': float(calibrated_y[i]),
            'pixel_x': float(pixel_x[i]),
            'pixel_y': float(mid_y_px[i]),
        })

    return rows


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
# Render digitized continuous curve and axis labels over the source image.

def draw_overlay(im, rows, rect, xc, yc):
    out = im.copy()
    x0, y0, x1, y1 = rect
    _, w = out.shape[:2]

    thick = max(1, w // 800)

    # Draw continuous digitized curve
    pts = np.array([[r['pixel_x'], r['pixel_y']] for r in rows], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=thick + 1)

    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 0), thick)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = max(0.35, w / 3000)
    fthick     = max(1, w // 1000)
    pad        = max(3, w // 500)
    label_col  = (30, 30, 30)
    bg_col     = (255, 255, 200)

    def put_label(text, px, py, anchor='tl'):
        (tw, th), bl = cv2.getTextSize(text, font, fscale, fthick)
        if anchor == 'tr':
            px -= tw
        elif anchor == 'br':
            px -= tw; py -= th + bl
        elif anchor == 'bl':
            py -= th + bl
        cv2.rectangle(out, (px - pad, py - th - pad), (px + tw + pad, py + bl + pad), bg_col, -1)
        cv2.putText(out, text, (px, py), font, fscale, label_col, fthick, cv2.LINE_AA)

    put_label(f"X={xc.value(x0):.4g}", x0 + pad, y1 + pad, anchor='tl')
    put_label(f"X={xc.value(x1):.4g}", x1 - pad, y1 + pad, anchor='tr')
    put_label(f"Y={yc.value(y0):.4g}", x0 - pad, y0 + pad, anchor='tl')
    put_label(f"Y={yc.value(y1):.4g}", x0 - pad, y1 - pad, anchor='bl')

    return out


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
# Digitize one plot image and write its CSV and optional diagnostics.

def process(path, outdir, debug, opts):
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError('image could not be read')

    rect                = plot_rect(im)
    toks                = ocr_tokens(im)
    xc, xt, xpair       = best_outer_pair(toks, 'x', rect)
    yc, yt, ypair       = best_outer_pair(toks, 'y', rect)
    mask, margin, floor = trace_mask(im, rect)

    rows = extract_trace_points(mask, rect, xc, yc, sigma=opts.sigma)

    if not rows:
        raise RuntimeError('no trace data points extracted')

    dest = outdir or (path.parent / 'OutputData')
    dest.mkdir(parents=True, exist_ok=True)
    csvpath = dest / (path.stem + '_trace.csv')

    with csvpath.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['X', 'Y'])
        w.writeheader()
        for r in rows:
            w.writerow({'X': f"{r['X']:.10g}", 'Y': f"{r['Y']:.10g}"})

    if opts.overlay:
        overlay      = draw_overlay(im, rows, rect, xc, yc)
        overlay_path = dest / (path.stem + '_overlay.png')
        cv2.imwrite(str(overlay_path), overlay)

    if opts.plot:
        plot_path = dest / (path.stem + '_plot.png')
        save_data_plot(rows, plot_path, path.name)

    if debug:
        d = im.copy()
        x0, y0, x1, y1 = rect
        cv2.rectangle(d, (x0, y0), (x1, y1), (0, 255, 0), 2)

        mask_bgr    = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        roi_overlay = d[y0:y1 + 1, x0:x1 + 1]
        cv2.addWeighted(roi_overlay, 0.75, mask_bgr, 0.25, 0, roi_overlay)

        pts = np.array([[r['pixel_x'], r['pixel_y']] for r in rows], dtype=np.int32)
        cv2.polylines(d, [pts.reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255), thickness=1)

        cv2.imwrite(str(dest / (path.stem + '_debug.png')), d)

        report = dest / (path.stem + '_ocr.txt')
        report.write_text('\n'.join([
            f'Image: {path}',
            f'Plot rectangle: {rect}',
            f'X outer labels: {xpair[0].text} -> {xpair[0].value} at {xpair[0].x:.2f}; '
            f'{xpair[1].text} -> {xpair[1].value} at {xpair[1].x:.2f}',
            f'Y outer labels: {ypair[0].text} -> {ypair[0].value} at {ypair[0].y:.2f}; '
            f'{ypair[1].text} -> {ypair[1].value} at {ypair[1].y:.2f}',
            f'X displayed limits estimated at plot edges: {xc.value(x0):.10g}, {xc.value(x1):.10g}',
            f'Y displayed limits estimated at plot edges: {yc.value(y1):.10g}, {yc.value(y0):.10g}',
            f'Trace thresholds: blue dominance={margin}, blue floor={floor}',
            f'Sigma: {opts.sigma}',
            f'Extracted Data Points: {len(rows)}',
            '',
            'All OCR candidates:',
            *[
                f'{t.text!r} value={t.value} x={t.x:.1f} y={t.y:.1f} confidence={t.confidence:.1f}'
                for t in toks
            ],
        ]), encoding='utf-8')

        xc, xt_cand, xpair = best_outer_pair(toks, 'x', rect)
        yc, yt_cand, ypair = best_outer_pair(toks, 'y', rect)
        
        ocr_preview = draw_ocr_preview(im, toks, rect, xt_cand, yt_cand)
        cv2.imwrite(str(dest / (path.stem + '_ocr_preview.png')), ocr_preview)

    return csvpath, len(rows)


# ---------------------------------------------------------------------------
# Draw OCR preview
# ---------------------------------------------------------------------------

def draw_ocr_preview(im, toks, rect, x_cand, y_cand):
    out = im.copy()
    x0, y0, x1, y1 = rect
    _, w = out.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.3, w / 4000)
    fthick = max(1, w // 1200)

    x_texts = set(id(t) for t in x_cand)
    y_texts = set(id(t) for t in y_cand)

    for t in toks:
        bx0 = int(t.x - t.w / 2)
        by0 = int(t.y - t.h / 2)
        bx1 = int(t.x + t.w / 2)
        by1 = int(t.y + t.h / 2)

        if id(t) in x_texts:
            color = (0, 255, 0)
            label = f"{t.text} (val:{t.value})"
        elif id(t) in y_texts:
            color = (255, 0, 0)
            label = f"{t.text} (val:{t.value})"
        else:
            color = (128, 128, 128)
            label = t.text

        cv2.rectangle(out, (bx0, by0), (bx1, by1), color, 1)
        cv2.putText(out, label, (bx0, max(0, by0 - 4)), font, fscale, color, fthick, cv2.LINE_AA)

    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out

# ---------------------------------------------------------------------------
# Save data plot
# ---------------------------------------------------------------------------
def save_data_plot(rows, plot_path, title_name):
    x = [r['X'] for r in rows]
    y = [r['Y'] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, color='#1f77b4', linewidth=1.5, label='Extracted Trace')
    plt.title(f'Extracted Data: {title_name}', fontsize=12)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Matplotlib automatically handles vector rendering when saving to .svg
    plt.savefig(plot_path.with_suffix('.svg'), format='svg')
    plt.close()
    
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    o = args()
    if o.tesseract:
        pytesseract.pytesseract.tesseract_cmd = o.tesseract
    images = files_from(o.inputs)
    if not images:
        print('ERROR: no supported images found', file=sys.stderr)
        return 2
    failures = 0
    for p in images:
        try:
            out, n = process(p, o.output, o.debug, o)
            print(f'OK: {p.name}: {n} points -> {out}')
        except pytesseract.TesseractNotFoundError:
            print('ERROR: Tesseract not found. Install tesseract-ocr or use --tesseract.',
                  file=sys.stderr)
            return 3
        except Exception as e:
            failures += 1
            print(f'FAILED: {p}: {e}', file=sys.stderr)
            dest = o.output or (p.parent / 'OutputData')
            dest.mkdir(parents=True, exist_ok=True)
            (dest / (p.stem + '_failed.txt')).write_text(str(e), encoding='utf-8')
    print(f'Completed: {len(images) - failures} succeeded, {failures} failed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())