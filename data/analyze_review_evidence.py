from pathlib import Path
import argparse
import csv, json
from collections import Counter, defaultdict
import numpy as np

root = Path.cwd()
final = root/'results/04_final_comparison'
parser = argparse.ArgumentParser(description="Summarize the saved review evidence from the project root.")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
model_keys = ('lr','cnn','cnn_optimized')
arms = ('clean','augmented')
seeds = (42,43,44)
rates = (0.,.05,.1,.20)

def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def predictions(path):
    result = defaultdict(dict)
    for r in csv_rows(path):
        view = result[float(r['noise_rate'])]
        assert r['source_id'] not in view
        view[r['source_id']] = (int(r['label']),int(r['prediction']))
    return result

def accuracy(rows):
    return 100*sum(y == pred for y,pred in rows.values())/len(rows)

heldout = []
for model in model_keys:
    for arm in arms:
        p = predictions(final/'confirmation'/arm/f'{model}_predictions.csv')
        assert set(p) == {0.,.1} and all(len(x) == 4000 for x in p.values())
        assert {k:v[0] for k,v in p[0.].items()} == {k:v[0] for k,v in p[.1].items()}
        heldout.append(dict(model=model,arm=arm,clean=accuracy(p[0.]),noise10=accuracy(p[.1])))

p = predictions(final/'seed_42/clean/cnn_optimized_predictions.csv')
ids = list(p[0.]); labels = np.array([p[0.][sid][0] for sid in ids])
assert len(ids)==7600 and np.bincount(labels).tolist()==[1900]*4

def correctness(p, rate):
    assert set(p[rate])==set(ids)
    assert np.array_equal(labels,[p[rate][sid][0] for sid in ids])
    return np.array([p[rate][sid][0]==p[rate][sid][1] for sid in ids],dtype=np.int8)

# Resample the same articles for both arms and all fitting seeds, separately within class.
# The estimate averages the per-article correctness differences across the three fits first.
rng = np.random.default_rng(92123)
indices = [rng.choice(np.flatnonzero(labels==label), size=(2000,1900), replace=True) for label in range(4)]
contrasts = []
original_20 = []
for model in model_keys:
    model_dir = final if model == 'cnn_optimized' else root/'results/01_baselines'
    pairs = [(predictions(model_dir/f'seed_{seed}'/'clean'/f'{model}_predictions.csv'),
              predictions(model_dir/f'seed_{seed}'/'augmented'/f'{model}_predictions.csv')) for seed in seeds]
    for rate in rates:
        delta = np.array([correctness(aug,rate)-correctness(clean,rate) for clean,aug in pairs])
        article_delta = delta.mean(axis=0)
        boot = sum(article_delta[take].mean(axis=1) for take in indices)/4
        lo,hi = 100*np.quantile(boot,[.025,.975])
        contrasts.append(dict(model=model,rate=rate,mean_pp=100*article_delta.mean(),low_pp=lo,high_pp=hi,
                              by_seed_pp={str(seed): 100*d.mean() for seed,d in zip(seeds,delta)}))
        if rate==.2:
            for seed,(clean,aug) in zip(seeds,pairs):
                for arm, pred in zip(arms,(clean,aug)):
                    original_20.append(dict(model=model,seed=seed,arm=arm,accuracy_pct=accuracy(pred[rate])))

confusions = {}
for rate in (0.,.2):
    matrix = np.zeros((4,4),dtype=int)
    for true,pred in p[rate].values():
        matrix[true,pred] += 1
    assert matrix.sum(axis=1).tolist()==[1900]*4
    confusions[str(rate)] = matrix.tolist()

training = []
rows = csv_rows(root/'report/data/metrics.csv')
for model in model_keys:
    use = [r for r in rows if r['model']==model and float(r['noise_rate'])==0.]
    assert len(use)==6
    training.append(dict(model=model,records=[dict(seed=int(float(r['seed'])),arm=r['regime'],
        seconds=float(r['training_seconds']), epochs=int(float(r['epochs_run'])) if r['epochs_run'] else None,
        parameters=int(float(r['parameters'])) if r['parameters'] else 4*(int(float(r['features']))+1)) for r in use]))

learning = []
for arm in arms:
    paths = [root/'results/03_cnn_development/seed_42'/arm/f'cnn_{kind}_history.json' for kind in ('recipe','subword')]
    scheduled,subword = [json.loads(path.read_text()) for path in paths]
    assert all(a['epoch']==b['epoch'] for a,b in zip(scheduled,subword))
    learning.append(dict(arm=arm, first_lr_reduction={kind:next((r['epoch'] for r in h if r['learning_rate']<.001),None)
        for kind,h in zip(('scheduled','subword'),(scheduled,subword))},
        early_deltas_pp={str(a['epoch']):100*(b['validation_objective']-a['validation_objective'])
                         for a,b in zip(scheduled[:8],subword[:8])}))

out = dict(heldout=heldout, bootstrap=dict(resamples=2000,seed=92123, contrasts=contrasts),
           original_20=original_20, confusions=confusions, training=training, learning=learning)
out['transitions'] = {f'{int(a)}->{int(b)}': int(n) for (a,b), n in Counter(zip(correctness(p,0.),correctness(p,.2))).items()}
args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
