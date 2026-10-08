"""Regenerate README visualizations from tracked archived experiment snapshots.

Run from repository root: python assets/readme/render_plots.py
Requires: numpy, pandas, matplotlib.

No benchmark experiments are executed by this script.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / 'benchmark_snapshot.csv'
DARK = '#152b42'
SLATE = '#54708b'
TEAL = '#167e8d'
BLUE = '#2673a6'
AMBER = '#c8754e'
PAPER = '#ecf0f3'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.titlesize': 13, 'axes.labelsize': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#cbd5df', 'axes.labelcolor': DARK,
    'text.color': DARK, 'xtick.color': SLATE, 'ytick.color': SLATE,
    'figure.facecolor': 'white', 'axes.facecolor':'white',
    'savefig.facecolor':'white', 'savefig.bbox':'tight', 'savefig.pad_inches':0.16,
    'grid.color': '#e9eef2', 'grid.linewidth': 0.8,
})

frame = pd.read_csv(CSV)
frame['dataset'] = pd.Categorical(frame['dataset'], categories=['exchange', 'electricity', 'solar', 'traffic', 'taxi'], ordered=True)
frame=frame.sort_values('dataset').reset_index(drop=True)
name = [s.title() for s in frame.dataset.astype(str)]

# Figure 1: normalized comparison, not pooled error bars from incomparable estimators.
fig, ax = plt.subplots(figsize=(10.6, 4.6))
ratios = (frame.archive_crps_sum / frame.paper_crps_sum_mean).to_numpy()
y = np.arange(len(frame))
colors=[TEAL if v <= 1.0 else AMBER for v in ratios]
ax.barh(y, ratios, height=0.58, color=colors, alpha=0.92)
ax.axvline(1, color=DARK, linewidth=1.5, linestyle=(0,(4,3)), label='Paper TimeGrad mean (10 runs)')
for i,(r,a,p) in enumerate(zip(ratios,frame.archive_crps_sum,frame.paper_crps_sum_mean)):
    ax.text(r + 0.02, i, f'{r:.2f}×', va='center', fontsize=10, fontweight='bold', color=DARK)
    ax.text(0.02, i, f'{a:.4f} / {p:.4f}', va='center', color='white' if r>0.72 else DARK, fontsize=8.3, fontweight='bold')
ax.set_yticks(y, name)
ax.invert_yaxis()
ax.set_xlim(0,1.52)
ax.set_xlabel('Archived single-run CRPS$_{sum}$ / paper 10-run mean')
ax.set_title('Benchmark comparison  |  five reproduced datasets', loc='left', pad=16, fontweight='bold')
# The dashed baseline is explained by the x-axis and README caption.
ax.grid(axis='x'); ax.set_axisbelow(True)
fig.text(0.125, -0.025, 'Bars left of 1.0 indicate a lower recorded score. Values are descriptive, not matched-seed comparisons.',
         fontsize=8.5, color=SLATE)
fig.savefig(HERE/'benchmark_comparison.png', dpi=200)
plt.close(fig)

# Figure 2: distinguish marginal interval coverage from aggregate CRPS.
fig, axes=plt.subplots(1,2,figsize=(11.0,4.7),sharey=True, gridspec_kw={'wspace':0.16})
for ax,col,nominal,color,title in zip(axes,['coverage_50','coverage_90'],[.5,.9],[TEAL,BLUE],['50% central interval','90% central interval']):
    x=frame[col].to_numpy()
    ax.axvline(nominal, color=SLATE, ls=(0,(4,3)), lw=1.35, label=f'Nominal {int(nominal*100)}%')
    for i,value in enumerate(x):
        ax.plot([min(value,nominal),max(value,nominal)],[i,i],color='#dbe4ec',lw=4,solid_capstyle='round', zorder=1)
    ax.scatter(x,y,s=74,color=color,edgecolor='white',linewidth=1.0,zorder=3,label='Empirical coverage')
    for i,value in enumerate(x):
        ax.text(value+0.023,i,f'{value*100:.1f}%',va='center',fontsize=8.5,color=DARK)
    ax.set_xlim(0,1.06); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(['0%','25%','50%','75%','100%'])
    ax.set_title(title,loc='left',pad=12,fontweight='bold')
    ax.grid(axis='x'); ax.set_axisbelow(True)
axes[0].set_yticks(y,name); axes[0].invert_yaxis()
axes[1].tick_params(axis='y',labelleft=False)
fig.suptitle('Marginal forecast-interval calibration  |  archived run',fontsize=14,weight='bold',x=.125,y=1.03,ha='left')
fig.text(.125,-.025,'Coverage is computed across dimensions, forecast steps and evaluation windows; it is not a joint coverage metric.',
         fontsize=8.5,color=SLATE)
fig.savefig(HERE/'interval_coverage.png',dpi=200)
plt.close(fig)

# Figure 3: mathematical *schedule*, not a learned artifact.
N=100
betas=np.linspace(1e-4,0.1,N)
abar=np.cumprod(1-betas)
fig, axes=plt.subplots(1,2,figsize=(11.0,3.8),gridspec_kw={'wspace':.23})
axes[0].plot(np.arange(1,N+1),betas,color=BLUE,lw=2.6)
axes[0].fill_between(np.arange(1,N+1),betas,color=BLUE,alpha=.1)
axes[0].set_title('Linear variance schedule',loc='left',fontweight='bold',pad=12)
axes[0].set_xlabel('Diffusion step $n$');axes[0].set_ylabel('$\\beta_n$')
axes[0].set_xlim(1,N);axes[0].grid(alpha=.7)
axes[1].plot(np.arange(1,N+1),abar,color=TEAL,lw=2.5,label='Signal weight $\\bar{\\alpha}_n$')
axes[1].plot(np.arange(1,N+1),1-abar,color=AMBER,lw=2.5,label='Noise weight $1-\\bar{\\alpha}_n$')
axes[1].set_title('Cumulative forward perturbation',loc='left',fontweight='bold',pad=12)
axes[1].set_xlabel('Diffusion step $n$');axes[1].set_ylabel('Weight in $q(x^n\\mid x^0)$')
axes[1].set_ylim(0,1.02);axes[1].set_xlim(1,N);axes[1].grid(alpha=.7)
axes[1].legend(loc='center right',fontsize=8,frameon=False)
fig.suptitle('Fixed forward diffusion process  |  N = 100',fontsize=14,weight='bold',x=.125,y=1.10,ha='left')
fig.savefig(HERE/'diffusion_schedule.png',dpi=200)
plt.close(fig)

# Figure 4: archived tuning histories only; archived full-data phase isn't present in currently active train.py.
fig, axes=plt.subplots(1,5,figsize=(15,3.5),sharey=False,gridspec_kw={'wspace':.34})
for i,(ax,dataset) in enumerate(zip(axes,frame.dataset.astype(str))):
    hist=pd.read_csv(HERE/'data'/f'{dataset}_training_history.csv')
    tune=hist[hist.phase=='tune']
    ax.plot(tune.epoch,tune.train_loss,label='Train',lw=1.9,color=BLUE)
    ax.plot(tune.epoch,tune.validation_loss,label='Validation',lw=1.8,color=AMBER)
    ax.set_title(dataset.title(),weight='bold',loc='left',pad=9)
    ax.set_xlabel('Epoch'); ax.set_yscale('log')
    ax.grid(True,which='major',alpha=.65)
    if i==0: ax.set_ylabel('Noise-prediction MSE (log)')
fig.legend(['Training','Validation'],loc='upper right',bbox_to_anchor=(.90,1.10),ncol=2,frameon=False)
fig.suptitle('Archived model-selection phase  |  training traces',fontsize=14,fontweight='bold',x=.125,y=1.13,ha='left')
fig.text(.125,-.07,'From included training_history.csv files (phase = tune); not a claim about the currently active train.py refit path.',fontsize=8.3,color=SLATE)
fig.savefig(HERE/'tuning_curves.png',dpi=200)
plt.close(fig)
print('Generated charts:',*[p.name for p in HERE.glob('*.png')],sep='\n- ')
