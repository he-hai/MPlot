# %% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit.models import GaussianModel
from lmfit import Parameters

plt.rcParams["font.family"] = "Arial"
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.fontsize'] = 13
plt.rcParams.update({'mathtext.default': 'regular'})

params = pd.DataFrame(columns=['center','sigma','amplitude'])
# %% 
## ==*== users input ==*==
title = 'Demo'
file_name = 'eventsFitted_Demo'
fig_save = None #'png'  # None, png, eps
bin_width = 0 # 0 -> use default 6; otherwise defining here
x_max = np.inf  # right limit for x-axis: np.inf takes from the dataset

peak_n = 1  # int: 0 -> histplot only; 1~N
# initial parameters: center_[value, min, max], sigma, amplitude
params.loc[1,:]=[[94,70,130],10,1000]
# params.loc[2,:]=[[47,30,70],10,1000]
# ...
## ======================================
# %% [Main Function]
df = pd.read_csv(f'{file_name}.csv',comment='#')
mw = df[df['masses_kDa']>0]['masses_kDa']

mw_max = np.ceil(mw.max()/10)*10
if x_max != np.inf:
    mw_max = x_max

if bin_width == 0:
    bin_width = 6
bins = np.arange(0,mw_max, bin_width)

counts = mw.value_counts(bins=bins)
x = counts.index.mid
y = counts.to_numpy()

gmods = [GaussianModel]*peak_n
def MultiGauss(peak, pars, params=params):
    gmods[peak] = GaussianModel(prefix=f'g{peak+1}_')
    pars.update(gmods[peak].make_params())
    pars[f'g{peak+1}_center'].set(
        value=params.iloc[peak,0][0],
        min=params.iloc[peak,0][1],
        max=params.iloc[peak,0][2]
    )
    pars[f'g{peak+1}_sigma'].set(value=params.iloc[peak,1])
    pars[f'g{peak+1}_amplitude'].set(value=params.iloc[peak,2])

fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(
    data=mw, ax=ax, binwidth=bin_width,
    element='step',alpha=0.1,
)
plt.xlabel('Mass [kDa]')
plt.xlim((0,mw_max))
plt.ylabel('Counts')
plt.title(f'{title}')

if peak_n > 0:
    gmods[0] = GaussianModel(prefix='g1_')
    pars=gmods[0].make_params()
    pars['g1_center'].set(
        value=params.iloc[0,0][0],
        min=params.iloc[0,0][1],
        max=params.iloc[0,0][2]
    )
    pars['g1_sigma'].set(value=params.iloc[0,1])
    pars['g1_amplitude'].set(value=params.iloc[0,2])
    if peak_n >= 1:
        for i in np.arange(0,peak_n):
            MultiGauss(i,pars,)
        model = np.sum(gmods)
        out = model.fit(y, pars, x=x)
        # print(out.fit_report())

        for i in np.arange(0,peak_n):
            c = out.values[f'g{i+1}_center']
            sigma = out.values[f'g{i+1}_sigma']
            x_ = np.arange(c-3*sigma,c+3*sigma)
            comps=out.eval_components(x=x_)

            # index_ = (
            #     (counts.index.left >= c-3*sigma-bin_width) & 
            #     (counts.index.right <= c+3*sigma+bin_width)
            # )
            # p = counts[index_].sum()/counts.sum()

            p = mw[(mw>c-3*sigma) & (mw<c+3*sigma)].count()/mw.count()
            sns.lineplot(
                x=x_,y=comps[f'g{i+1}_'],
                ax=ax,ls='--',lw=2,
                label=f"MW={c:.0f}, $\sigma$={sigma:.0f}, {p:.0%}"
            )

# sns.lineplot(x=x,y=out.best_fit,ax=ax,color='k',lw=2)

if fig_save:
    fig.savefig(f'{title}_MP.{fig_save}',bbox='tight')
plt.show()
