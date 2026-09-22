#!/usr/bin/env python3
"""Linux/Windows plot extrema digitizer, version 2.

Extracts visible local maxima and minima from one blue trace, calibrates linear
axes with OCR, and writes CSV. Supports one image, multiple images, or folders.

Linux setup (Debian/Ubuntu):
  sudo apt install tesseract-ocr python3-pip
  python3 -m pip install opencv-python numpy scipy pytesseract

Examples:
  python3 plot_extrema_digitizer_v2.py image.png --debug
  python3 plot_extrema_digitizer_v2.py ./plots --debug
  python3 plot_extrema_digitizer_v2.py 'plots/*.png' -o ./csv
"""
from __future__ import annotations
import argparse, csv, glob, re, sys
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from scipy.signal import find_peaks

EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}

@dataclass
class Token:
    value: float; x: float; y: float; w: float; h: float; text: str; confidence: float

@dataclass
class Calibration:
    slope: float; intercept: float; low_value: float; high_value: float
    low_pixel: float; high_pixel: float
    def value(self,p): return self.slope*np.asarray(p)+self.intercept

def args():
    p=argparse.ArgumentParser(description='Extract visible local maxima/minima from plot images.')
    p.add_argument('inputs',nargs='+',help='Images, folders, or glob patterns')
    p.add_argument('-o','--output',type=Path,help='Output directory (default: beside each image)')
    p.add_argument('--debug',action='store_true',help='Write annotated PNG and OCR report')
    p.add_argument('--prominence',type=float,default=0.0,help='Minimum prominence in Y units; 0=automatic')
    p.add_argument('--min-distance',type=int,default=3,help='Minimum extrema spacing in pixels')
    p.add_argument('--trace-color',choices=['auto','blue'],default='auto')
    p.add_argument('--tesseract',help='Optional full path to tesseract executable')
    return p.parse_args()

def files_from(items):
    out=[]
    for s in items:
        matches=[Path(x) for x in glob.glob(s)]
        if not matches: matches=[Path(s)]
        for p in matches:
            if p.is_dir(): out += sorted(x for x in p.iterdir() if x.suffix.lower() in EXTS)
            elif p.is_file() and p.suffix.lower() in EXTS: out.append(p)
    return list(dict.fromkeys(x.resolve() for x in out))

def plot_rect(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    # Plot interiors from the source application are broad light-gray regions.
    light=((hsv[:,:,1] < 55) & (hsv[:,:,2] >= 145) & (hsv[:,:,2] <= 245))
    row_ok=light.mean(axis=1) > .55
    col_ok=light.mean(axis=0) > .55
    def longest(mask):
        d=np.diff(np.r_[False,mask,False].astype(np.int8))
        starts=list(np.where(d==1)[0]); ends=list(np.where(d==-1)[0]-1)
        if not starts: return None
        # Merge narrow gaps caused by solid cursor/reference lines crossing the plot.
        ms=[starts[0]]; me=[]; current=ends[0]
        for st,en in zip(starts[1:],ends[1:]):
            if st-current-1 <= 10: current=en
            else: me.append(current); ms.append(st); current=en
        me.append(current)
        lengths=np.asarray(me)-np.asarray(ms)
        i=int(np.argmax(lengths)); return int(ms[i]),int(me[i])
    yr=longest(row_ok); xr=longest(col_ok)
    if yr and xr and (xr[1]-xr[0])>.60*im.shape[1] and (yr[1]-yr[0])>.50*im.shape[0]:
        return xr[0],yr[0],xr[1],yr[1]
    raise RuntimeError('plot rectangle not found')

def num(s):
    s=s.strip().replace('−','-').replace(',','.').replace('O','0').replace('o','0')
    s=re.sub(r'[^0-9+\-.]','',s)
    if s in ('','-','+','.','-.','+.'): return None
    if s.startswith('.'): s='0'+s
    if s.startswith('-.'): s='-0'+s[1:]
    try: return float(s)
    except ValueError: return None

def ocr_tokens(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); scale=3
    big=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    tokens=[]; cfg='--psm 11 -c tessedit_char_whitelist=0123456789.-+'
    for a in (big,cv2.bitwise_not(big)):
        d=pytesseract.image_to_data(a,config=cfg,output_type=pytesseract.Output.DICT)
        for i,t in enumerate(d['text']):
            v=num(t)
            try: conf=float(d['conf'][i])
            except: conf=-1
            if v is None or conf<5: continue
            tokens.append(Token(v,(d['left'][i]+d['width'][i]/2)/scale,
                                (d['top'][i]+d['height'][i]/2)/scale,
                                d['width'][i]/scale,d['height'][i]/scale,t,conf))
    # Deduplicate two OCR polarities.
    keep=[]
    for t in sorted(tokens,key=lambda z:z.confidence,reverse=True):
        if not any(abs(t.value-q.value)<1e-12 and abs(t.x-q.x)<7 and abs(t.y-q.y)<7 for q in keep): keep.append(t)
    return keep

def best_outer_pair(tokens,axis,rect):
    x0,y0,x1,y1=rect
    if axis=='x':
        cand=[t for t in tokens if x0-15<=t.x<=x1+15 and y1-12<=t.y<=y1+75]
        pix=lambda t:t.x; sign=1
    else:
        cand=[t for t in tokens if x0-115<=t.x<=x0+18 and y0-12<=t.y<=y1+12]
        pix=lambda t:t.y; sign=-1
    if len(cand)<2: raise RuntimeError(f'OCR found fewer than two usable {axis}-axis labels')
    # Tesseract frequently drops the leading decimal point from labels such as '.12'.
    # When every X token is a 2-digit integer, interpret the family as hundredths.
    if axis=='x' and len(cand)>=2 and all(float(t.value).is_integer() and 10<=abs(t.value)<=99 for t in cand):
        cand=[Token(t.value/100.0,t.x,t.y,t.w,t.h,t.text,t.confidence) for t in cand]
    span=(x1-x0) if axis=='x' else (y1-y0)
    pairs=[]
    for a in cand:
      for b in cand:
        dp=pix(b)-pix(a); dv=b.value-a.value
        if dp<=max(12,.18*span) or sign*dv<=0: continue
        coverage=dp/span
        # Prefer outer labels, high confidence, and decimal X values when present.
        plaus=coverage + .002*(a.confidence+b.confidence)
        if axis=='x' and max(abs(a.value),abs(b.value))>1000: plaus-=2
        pairs.append((plaus,a,b))
    if not pairs: raise RuntimeError(f'no consistent {axis}-axis outer-label pair')
    _,a,b=max(pairs,key=lambda z:z[0])
    slope=(b.value-a.value)/(pix(b)-pix(a)); intercept=a.value-slope*pix(a)
    # Sanity: scale must be finite, monotonic, and cover a meaningful part of the plot.
    if not np.isfinite(slope) or slope==0: raise RuntimeError(f'invalid {axis}-axis scale')
    vals=[a.value,b.value]
    return Calibration(slope,intercept,min(vals),max(vals),pix(a),pix(b)),cand,(a,b)

def trace_mask(im,rect):
    x0,y0,x1,y1=rect; roi=im[y0:y1+1,x0:x1+1]; b,g,r=cv2.split(roi)
    B=b.astype(np.int16); G=g.astype(np.int16); R=r.astype(np.int16)
    dominance=B-np.maximum(G,R)
    # Adaptive threshold derived from strongly blue pixels, bounded for stability.
    positive=dominance[dominance>10]
    margin=int(np.clip(np.percentile(positive,35) if positive.size else 25,15,70))
    floor=int(np.clip(np.percentile(B[dominance>margin],10) if np.any(dominance>margin) else 90,70,180))
    m=((dominance>=margin)&(B>=floor)).astype(np.uint8)*255
    return m,margin,floor

def interp(a,max_gap=4):
    a=a.astype(float).copy(); ok=np.isfinite(a)
    if ok.sum()<2:return a
    idx=np.arange(len(a)); estimate=np.interp(idx,idx[ok],a[ok]); missing=~ok
    changes=np.diff(np.r_[False,missing,False].astype(np.int8))
    for s,e in zip(np.where(changes==1)[0],np.where(changes==-1)[0]-1):
        if e-s+1<=max_gap:a[s:e+1]=estimate[s:e+1]
    return a

def parabola_refine(y,i,mode):
    if i<=0 or i>=len(y)-1 or not np.all(np.isfinite(y[i-1:i+2])): return float(i),float(y[i])
    yy=y[i-1:i+2] if mode=='max' else -y[i-1:i+2]
    den=yy[0]-2*yy[1]+yy[2]
    if abs(den)<1e-12:return float(i),float(y[i])
    delta=float(np.clip(.5*(yy[0]-yy[2])/den,-.5,.5))
    vertex=yy[1]-.25*(yy[0]-yy[2])*delta
    return i+delta,float(vertex if mode=='max' else -vertex)

def extrema(mask,rect,xc,yc,min_distance,prominence):
    x0,y0,_,_=rect; w=mask.shape[1]
    top=np.full(w,np.nan); bottom=np.full(w,np.nan)
    for i in range(w):
        rows=np.flatnonzero(mask[:,i])
        if rows.size: top[i]=rows.min()+y0; bottom[i]=rows.max()+y0
    top,bottom=interp(top),interp(bottom)
    if np.isfinite(top).sum()<10: raise RuntimeError('too few trace pixels detected')
    def fill(a):
        ok=np.isfinite(a); return np.interp(np.arange(len(a)),np.flatnonzero(ok),a[ok])
    upper=yc.value(fill(top)); lower=yc.value(fill(bottom))
    yspan=abs(yc.value(rect[3])-yc.value(rect[1]))
    prom=prominence if prominence>0 else max(yspan*.005,abs(yc.slope)*2)
    imax,pm=find_peaks(upper,distance=max(1,min_distance),prominence=prom)
    imin,pn=find_peaks(-lower,distance=max(1,min_distance),prominence=prom)
    rows=[]
    for ids,series,kind,props in ((imax,upper,'Max',pm),(imin,lower,'Min',pn)):
        for j,i in enumerate(ids):
            px,vy=parabola_refine(series,int(i),'max' if kind=='Max' else 'min')
            rows.append({'X':float(xc.value(px+x0)),'Y':vy,'Type':kind,
                         'Prominence':float(props['prominences'][j]),'pixel_x':px+x0,
                         'pixel_y':float((vy-yc.intercept)/yc.slope)})
    return sorted(rows,key=lambda z:(z['X'],z['Type'])),prom

def process(path,outdir,debug,opts):
    im=cv2.imread(str(path));
    if im is None: raise RuntimeError('image could not be read')
    rect=plot_rect(im); toks=ocr_tokens(im)
    xc,xt,xpair=best_outer_pair(toks,'x',rect); yc,yt,ypair=best_outer_pair(toks,'y',rect)
    mask,margin,floor=trace_mask(im,rect)
    rows,prom=extrema(mask,rect,xc,yc,opts.min_distance,opts.prominence)
    if not rows: raise RuntimeError('no extrema met the selected settings')
    dest=(outdir or path.parent); dest.mkdir(parents=True,exist_ok=True)
    csvpath=dest/(path.stem+'_extrema.csv')
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['X','Y','Type','Prominence']); w.writeheader()
        for r in rows:w.writerow({k:f'{r[k]:.10g}' if k!='Type' else r[k] for k in w.fieldnames})
    if debug:
        d=im.copy(); x0,y0,x1,y1=rect; cv2.rectangle(d,(x0,y0),(x1,y1),(0,255,0),2)
        for r in rows: cv2.circle(d,(round(r['pixel_x']),round(r['pixel_y'])),3,(0,0,255) if r['Type']=='Max' else (0,165,255),-1)
        cv2.imwrite(str(dest/(path.stem+'_debug.png')),d)
        report=dest/(path.stem+'_ocr.txt')
        report.write_text('\n'.join([
          f'Image: {path}',f'Plot rectangle: {rect}',
          f'X outer labels: {xpair[0].text} -> {xpair[0].value} at {xpair[0].x:.2f}; {xpair[1].text} -> {xpair[1].value} at {xpair[1].x:.2f}',
          f'Y outer labels: {ypair[0].text} -> {ypair[0].value} at {ypair[0].y:.2f}; {ypair[1].text} -> {ypair[1].value} at {ypair[1].y:.2f}',
          f'X displayed limits estimated at plot edges: {xc.value(x0):.10g}, {xc.value(x1):.10g}',
          f'Y displayed limits estimated at plot edges: {yc.value(y1):.10g}, {yc.value(y0):.10g}',
          f'Trace thresholds: blue dominance={margin}, blue floor={floor}',f'Prominence used: {prom:.10g}',f'Extrema: {len(rows)}',
          '', 'All OCR candidates:', *[f'{t.text!r} value={t.value} x={t.x:.1f} y={t.y:.1f} confidence={t.confidence:.1f}' for t in toks]
        ]),encoding='utf-8')
    return csvpath,len(rows)

def main():
    o=args()
    if o.tesseract:pytesseract.pytesseract.tesseract_cmd=o.tesseract
    images=files_from(o.inputs)
    if not images: print('ERROR: no supported images found',file=sys.stderr); return 2
    failures=0
    for p in images:
        try:
            out,n=process(p,o.output,o.debug,o); print(f'OK: {p.name}: {n} extrema -> {out}')
        except pytesseract.TesseractNotFoundError:
            print('ERROR: Tesseract not found. Install tesseract-ocr or use --tesseract.',file=sys.stderr); return 3
        except Exception as e:
            failures+=1; print(f'FAILED: {p}: {e}',file=sys.stderr)
            # Continue batch processing instead of terminating.
            dest=o.output or p.parent; dest.mkdir(parents=True,exist_ok=True)
            (dest/(p.stem+'_failed.txt')).write_text(str(e),encoding='utf-8')
    print(f'Completed: {len(images)-failures} succeeded, {failures} failed')
    return 1 if failures else 0
if __name__=='__main__': raise SystemExit(main())
