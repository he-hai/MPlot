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

def MultiGauss(peak_n, params):
    gmods = [GaussianModel]*peak_n
    for i in np.arange(0,peak_n):
        gmods[i] = GaussianModel(prefix=f'g{i+1}_')

    pars=gmods[0].make_params()
    for i in np.arange(0,peak_n):
        pars.update(gmods[i].make_params())
        pars[f'g{i+1}_center'].set(
            value=params.iloc[i,0][0],
            min=params.iloc[i,0][1],
            max=params.iloc[i,0][2]
        )
        pars[f'g{i+1}_sigma'].set(value=params.iloc[i,1])
        pars[f'g{i+1}_amplitude'].set(value=params.iloc[i,2])
    return gmods, pars

def MP_plot(
    peak_n: int,
    x_max: int,
    fig_save: str, 
    title: str,
    folder_name: str,
    params,
    bin_width: float=6,
):
    df = pd.read_csv(f'{folder_name}\\eventsFitted.csv',comment='#')
    mw = df[df['masses_kDa']>0]['masses_kDa']

    mw_max = np.ceil(mw.max()/10)*10
    if x_max != np.Infinity:
        mw_max = x_max

    bins = np.arange(0,mw_max, bin_width)

    counts = mw.value_counts(bins=bins)
    x = counts.index.mid
    y = counts.to_numpy()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(
        data=mw, ax=ax, binwidth=bin_width,
        element='step',alpha=0.1,
    )
    plt.xlabel('Mass [kDa]')
    plt.xlim((0,mw_max))
    plt.ylabel('Counts')
    plt.title(f'{title}')

    if peak_n != 0:
        (gmods, pars)=MultiGauss(peak_n,params)
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
# %%
## ==*== users input ==*==
title = 'Protein1'
folder_name = 'Protein1'

peak_n = 1  # int: 0 -> histplot only; 1~N
params = pd.DataFrame(columns=['center','sigma','amplitude'])
# initial parameters: center_[value, min, max], sigma, amplitude
params.loc[1,:]=[[73,60,100],10,1000]
# params.loc[2,:]=[[40,30,50],10,1000]
# ...

x_max = 500 # np.Infinity  # right limit for x-axis: np.Infinity takes from the dataset
fig_save = None #'png'  # None, png, eps
bin_width = 6   # default 6; otherwise defining here

# MP_plot(peak_n, x_max, fig_save, title, folder_name, params, bin_width)
MP_plot(peak_n,x_max,fig_save,title,folder_name,params,bin_width)

# %%
## ==*== users input ==*==
title = 'Protein2'
folder_name = 'Protein2_1.5x'

peak_n = 2  # int: 0 -> histplot only; 1~N
params = pd.DataFrame(columns=['center','sigma','amplitude'])
# initial parameters: center_[value, min, max], sigma, amplitude
params.loc[1,:]=[[50,30,80],10,1000]
params.loc[2,:]=[[112,100,124],10,1000]
# ...

x_max = 500 # np.Infinity  # right limit for x-axis: np.Infinity takes from the dataset
fig_save = None #'png'  # None, png, eps
bin_width = 6  # default 6; otherwise defining here
# MP_plot(peak_n, x_max, fig_save, title, folder_name, params, bin_width)
MP_plot(peak_n,x_max,fig_save,title,folder_name,params,bin_width)