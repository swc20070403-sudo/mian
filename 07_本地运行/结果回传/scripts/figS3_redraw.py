"""Supplementary Fig. S3 redrawn WITHOUT GBDT: SOH-dependent mean bias and residual distribution of R0 (and A2).

Identical layout, bins, zoom window, KDE rule and output format as plotting/fig10_bias_vs_soh.py (its draw() body
is reused with the second series changed from GBDT to A2).  Data: the original clean-rerun test predictions of R0
and A2 (three-seed mean per record, 16 938 records).  Training venv (matplotlib 3.11.1, as the original figure).
"""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parents[1]))
from plotting import error_data, loaders
from plotting.qa import require_matplotlib_panel_alignment
from plotting.style import kde_bw
from plotting.style_errorfigs import BLUE, BROWN, GRAY, apply_style

OUT = WORK / 'deliverables'
NAME = 'figS3_bias_vs_soh_R0_A2'
ZOOM = (89, 97, -.65, .65)


def draw(data, name=NAME, models=('R0', 'A2')):
    apply_style()
    mpl.rcParams.update({'font.family':['Times New Roman','Arial'],
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.2047244094,2.9921259843),dpi=300)
    ax=fig.add_axes([.085,.215,.723,.755],label='overview')
    side=fig.add_axes([.837,.215,.148,.755],sharey=ax,label='marginal')
    resid=data['resid']
    # R0 lies inside the original y-range (-10, 7.5); A2 has 2 of 16 938 records outside it (-14.2, +7.6 pp),
    # which only clips the far tail of its marginal KDE -- the bin means are all inside.
    assert np.isfinite(resid['R0']).all() and resid['R0'].min()>-10 and resid['R0'].max()<7.5
    assert all(np.all((data['bins'][m][1]>-10)&(data['bins'][m][1]<7.5)) for m in models)
    for lo,hi in [(72,80),(95,100.8)]:ax.axvspan(lo,hi,color='#FBF2EA',lw=0,zorder=0)
    ax.axhline(0,color=GRAY,lw=.65,ls=(0,(3,2)),zorder=1)
    styles={'R0':dict(color=BLUE,ls='-',marker='o'),
            'A2':dict(color=BROWN,ls=(0,(4,2)),marker='s')}
    styles={k:v for k,v in styles.items() if k in models}
    for model,style in styles.items():
        x,y,n=data['bins'][model]
        assert len(x)==24 and np.all(np.diff(x)>0) and np.all(n>=20)
        ax.plot(x,y,**style,lw=1.15,ms=2.7,mfc='white',mew=.7,label=model,zorder=4)
    ax.set(xlim=(72,100.8),ylim=(-10,7.5),xticks=[75,80,85,90,95,100],yticks=[-8,-4,0,4],
           xlabel='Measured SOH (%)',ylabel='Residual (pp)')
    ax.legend(loc='upper right',ncol=len(models),handlelength=2,columnspacing=1.4,borderaxespad=.8)
    x0,x1,y0,y1=ZOOM
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#404850',lw=.65,zorder=5))
    zoom=ax.inset_axes([.07,.105,.46,.36],label='zoom')
    for model,style in styles.items():
        x,y,n=data['bins'][model]
        # Interpolate only intersections with inset borders, preserving the
        # existing straight segments; these are not new measured points.
        assert np.all(np.diff(x)>0) and x.min()<x0<x1<x.max()
        inside=(x>=x0)&(x<=x1)
        path_x=np.r_[x0,x[inside],x1]
        path_y=np.interp(path_x,x,y)
        assert path_y.min()>y0 and path_y.max()<y1
        zoom.plot(path_x,path_y,color=style['color'],ls=style['ls'],lw=1.05,zorder=3)
        zoom.plot(x[inside],y[inside],ls='',marker=style['marker'],color=style['color'],
                  ms=2.6,mfc='white',mew=.65,zorder=4)
    zoom.set(xlim=(x0,x1),ylim=(y0,y1),xticks=[89,91,93,95,97],yticks=[-.5,0,.5])
    zoom.tick_params(labelsize=6.5,length=2,pad=2)
    zoom.grid(axis='y',color='#E7EBEF',lw=.4)
    zoom.set_axisbelow(True)
    zoom.axhline(0,color=GRAY,lw=.55,ls=(0,(3,2)),zorder=1)
    connector=ConnectionPatch(xyA=(x0,y0),coordsA=ax.transData,
                               xyB=(1,1),coordsB=zoom.transAxes,
                               color='#68727B',lw=.55,zorder=2,clip_on=False)
    ax.add_artist(connector)
    grid=np.linspace(-10,7.5,500)
    density={};bandwidth={}
    for model,r in resid.items():
        if model not in models: continue
        q1,q3=np.quantile(r,[.25,.75]);bw=.9*min(r.std(),(q3-q1)/1.34)*r.size**(-1/5)
        assert bw>0
        bandwidth[model]=float(bw)
        density[model]=kde_bw(r,grid,bw)
    scale=max(d.max() for d in density.values())
    side.fill_betweenx(grid,0,density['R0']/scale,color=BLUE,alpha=.18,lw=0)
    side.plot(density['R0']/scale,grid,color=BLUE,lw=1.1)
    if 'A2' in models: side.plot(density['A2']/scale,grid,color=BROWN,lw=1,ls=(0,(4,2)))
    side.axhline(0,color=GRAY,lw=.65,ls=(0,(3,2)),zorder=0)
    side.set(xlim=(0,1.1),xticks=[0,.5,1],xlabel='Relative density')
    side.set_xticklabels(['0','0.5','1']);side.tick_params(axis='y',left=False,labelleft=False)
    require_matplotlib_panel_alignment(fig,axes=[ax,side])
    fig.savefig(OUT/(name+'.pdf'),dpi=600)
    fig.savefig(OUT/(name+'.svg'),dpi=600)
    fig.savefig(OUT/(name+'.png'),dpi=600)
    fig.savefig(OUT/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    return {'KDE_bandwidths_pp':bandwidth,'KDE_shared_normalization':float(scale)}



def load():
    p_r0, p_a2 = loaders.pooled('R0'), loaders.pooled('A2')
    if not np.array_equal(p_r0['row_id'], p_a2['row_id']):
        raise RuntimeError('R0 and A2 rows differ')
    true = p_r0['true_soh_percent']
    resid = {'R0': p_r0['pred_soh_percent'] - true, 'A2': p_a2['pred_soh_percent'] - true}
    assert true.size == 16938 and abs(np.sqrt(np.mean(resid['R0'] ** 2)) - 1.4608) < 5e-5
    edges = np.arange(np.floor(true.min()), 101.0, 1.0)
    bins = {k: error_data.residual_bins(true, r, edges) for k, r in resid.items()}
    with (OUT / 'figS3_bin_means.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'mean_measured_soh', 'mean_residual_pp', 'n_records'])
        for k, (x, m, n) in bins.items():
            for row in zip(x, m, n):
                w.writerow([k, *row])
    with (OUT / 'figS3_record_residuals.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['row_id', 'cell_id', 'rpt', 'measured_soh_percent', 'R0_residual_pp', 'A2_residual_pp'])
        for i in range(true.size):
            w.writerow([int(p_r0['row_id'][i]), int(p_r0['cell_id'][i]), int(p_r0['rpt'][i]), true[i], resid['R0'][i], resid['A2'][i]])
    return {'true': true, 'resid': resid, 'bins': bins}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    print(draw(data))
    print(draw(data, 'figS3_bias_vs_soh_R0_only', ('R0',)))
