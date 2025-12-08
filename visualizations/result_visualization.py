import json
import argparse
from collections import defaultdict
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def check_equality_of_run_hyperparams(l_d):

    def check_dict_equivalence(d1,d2):
        return d1 == d2

    dicts = [get_run_hyperparams(d) for d in l_d]

    uniques = [] 
    solved_equivalence_classes = defaultdict(list, [])

    for i,d1 in enumerate(dicts): 
        for j,d2 in uniques:
            if check_dict_equivalence(d1,d2):
                solved_equivalence_classes[j].append(i)
                break
        else:
            solved_equivalence_classes[i].append(i)
            uniques.append((i,d1))

    return solved_equivalence_classes.values()



def merge_dicts(l_d, savename=None):
    
    equiv_classes = check_equality_of_run_hyperparams(l_d)
    if len(equiv_classes) != 1:
        raise NotImplementedError("Dunno atm")
    else:
        res = {"benchmark_parameters": l_d[0]["benchmark_parameters"], 'runs': []}
        mix_dict = defaultdict(list, [])
        for d in l_d:
            for i in range(n_backends(d)):
                id = get_backend_id(d,i)
                mix_dict[id].append(get_backend_res(d,i))
        
        for i, (id,ds) in enumerate(mix_dict.items()):
            runs_element = {'runs': {}}

            vals = []
            for d in ds:
                vals.extend(zip(d['runs']['n'], d['runs']['price'], d['all_times'], d['runs']['time_ms_mean'], d['runs']['time_ms_std']))
            vals.sort()
            
            runs_element['do_pass_sanity_check'] = all([d['do_pass_sanity_check'] == 'true' for d in ds]) 
            runs_element['id'] = id
            runs_element['function_id'] = ds[0]['function_id']
            runs_element['hyperparams'] = ds[0]['hyperparams']
    
            runs_element['runs']['n'], runs_element['runs']['price'], runs_element['all_times'], runs_element['runs']['time_ms_mean'], runs_element['runs']['time_ms_std'] = zip(*vals)

            res['runs'].append(runs_element)
        
        if savename is not None:
            with open(savename, 'w') as f:
                json.dump(res, f, indent=2)

        return res




def get_run_hyperparams(d):
    d = deepcopy(d['benchmark_parameters'])
    del d["nrepetition_at_step"]
    del d["nstart"]
    del d["nend"]
    return d

def get_all_times(d, nidx, i):
    return d['runs'][i]['all_times'][nidx]

def get_sanity_check_bool(d, i):
     return d['runs'][i]['do_pass_sanity_check']

def get_function_id(d, i):
     return d['runs'][i]['function_id']

def get_backend_id(d, i):
     return d['runs'][i]['id']

def get_runs_dict(d, i):
     return d['runs'][i]['runs']

def get_backend_res(d, i):
     return d['runs'][i]

def get_preferred_name(d,i):
    return "_".join(d['runs'][i]['hyperparams'])  if len(d['runs'][i]['hyperparams']) else  "base"

def get_hyperparam(d, i):
    return d['runs'][i]['hyperparams']

def get_hyperparam_el(d, i, p):
    return d['runs'][i]['hyperparams'][p] if d['runs'][i]['hyperparams'] else -1

def n_backends(d):
    return len(d['runs'])

def n_depths(d):
    return len(d['runs'][0]['all_times'])

def get_n(d, n):
    return d['runs'][0]['runs']['n'][n]



def plot_tpb_split(res_dict):
    
    def ppprint(tpbdata, tpbfnames, tpbmeans, min_median, base_median, s): 
        norm = colors.Normalize(vmin=min(tpbmeans), vmax=max(tpbmeans))
        colormap = cm.get_cmap('RdYlGn_r')

        fig, ax = plt.subplots(figsize=(48,15))
        bp = ax.boxplot(tpbdata, tick_labels=tpbfnames, patch_artist=True, showfliers=False)

        for box, mean in zip(bp['boxes'], tpbmeans):
            box.set_facecolor(colormap(norm(mean)))

        ax.axhline(min_median, linestyle='--', linewidth=1, color=colormap(norm(min_median)), label="Minimum Median")
        ax.axhline(base_median, linestyle='--', linewidth=1, color=colormap(norm(base_median)), label="Base Median")


        plt.xticks(rotation=45, ha="right")

        plt.title(f"Boxplot for n = {get_n(res_dict, nidx)}")
        plt.xlabel("Backends")
        plt.ylabel("Time(ms)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/tpb/n{get_n(res_dict, nidx)}-TpB{TpB}{s}")


    for nidx in range(n_depths(res_dict)):
        idxs = list(range(n_backends(res_dict)))
        fnames = [get_preferred_name(res_dict, i) for i in idxs]
        data = [get_all_times(res_dict, nidx, i) for i in idxs]
        medians = [np.median(np.array(t)) for t in data]

        idxs, fnames, data, medians = zip(
            *sorted(zip(idxs, fnames, data, medians), key=lambda x: int(get_hyperparam_el(res_dict, x[0], 2)))
        )

        medians = np.array(medians)

        base_median = medians[fnames.index('base')]
        min_median = min(medians)

        for TpB in ['512', '256', '128']:
            selector = [i for i,(x,y) in enumerate(zip(medians, fnames)) if (TpB in y)]
            tpbdata = [x for i,x in enumerate(data) if i in selector]
            tpbfnames = np.array(fnames)[selector]
            tpbmedians = np.array(medians)[selector]

            if not len(tpbmedians): continue
            ppprint(tpbdata, tpbfnames, tpbmedians, min_median, base_median, '-all')

            selector = [i for i,(x,y) in enumerate(zip(medians, fnames)) if (x < np.quantile(tpbmedians, .25)) ]
            tpbdata = [x for i,x in enumerate(data) if i in selector]
            tpbfnames = np.array(fnames)[selector]
            tpbmedians = np.array(medians)[selector]

            if not len(tpbmedians): continue

            ppprint(tpbdata, tpbfnames, tpbmedians, min_median, base_median, '-q25')




def plot_evolution(res_dict):
    res_dict['runs'] = sorted(res_dict['runs'], key = lambda x: int(x['hyperparams'][2]) if x['hyperparams'] else 0)

    ns = [get_n(res_dict, nidx) for nidx in range(n_depths(res_dict))]
    min_medians = [min([np.median(np.array(get_all_times(res_dict, nidx, i))) for i in range(n_backends(res_dict))]) for nidx in range(n_depths(res_dict))]

    fig, ax = plt.subplots(figsize=(128, 100))
    ax.set_xscale('log')
    ax.set_yscale('log')

    colormap = cm.get_cmap('RdYlGn_r')
    norm = colors.Normalize(vmin=0, vmax=n_backends(res_dict))

    all_medians = [[] for _ in range(n_depths(res_dict))]
    
    for i in range(n_backends(res_dict)):
        data = [get_all_times(res_dict, nidx, i)/min_medians[nidx] for nidx in range(n_depths(res_dict))]
        fnames = get_preferred_name(res_dict, i)
        means = [np.mean(np.array(t)) for t in data]
        medians = [np.median(np.array(t)) for nidx,t in enumerate(data)]

        bp = ax.boxplot(
            data,
            positions=ns,
            widths=np.array(ns) * 0.05,
            manage_ticks=False,
        )

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=colormap(norm(i)))

        for nidx, m in enumerate(medians):
            all_medians[nidx].append((i, m))

        ax.plot(ns, medians, marker='o', linestyle='-', label=f"{fnames}", color=colormap(norm(i)))


    for nidx, medlist in enumerate(all_medians):
        sorted_meds = sorted(medlist, key=lambda x: x[1])
        topi = sorted_meds[:50]

        for i, m in topi:
            ax.text(
                ns[nidx],
                m,
                get_hyperparam(res_dict, i)[2] if get_hyperparam(res_dict, i) else "base",
                ha='center',
                va='bottom',
                fontsize=20,
                color='black'
            )


    ax.set_xlabel("n")
    ax.set_ylabel("Time")
    ax.set_ylim(.95, 1.5)
    ax.set_title("Timing vs n")
    ax.legend()

    plt.savefig(f"plots/evolution/all.png")



if __name__ == "__main__":

    dicts = []
    with open("../128-NStep125-100.json", 'r') as fh:    
        dicts.append(json.load(fh))

    with open("../128-NStepCM2-111.json", 'r') as fh:    
        dicts.append(json.load(fh))

    res_dict = merge_dicts(dicts, 'total-128.json')


    plot_tpb_split(res_dict)
    plot_evolution(res_dict)