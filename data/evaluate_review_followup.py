"""Re-evaluate frozen checkpoints for the report review. Run from the project root."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from project.data import Article, corrupt_corpus, metrics, write_csv, write_json
from project.experiments import FINAL, ROOT, read_jsonl, saved_model
from project.neural_models import load_cnn_variant, predict_cnn_variant


def score_views(predict, views, records, model, seed, arm, output):
    labels = [row['label'] for row in records]
    summaries, predictions = [], []
    for corruption_seed, rate, texts in views:
        pred = predict(texts)
        score = metrics(labels, pred)
        summaries.append(dict(model=model, seed=seed, arm=arm, corruption_seed=corruption_seed,
                              rate=rate, accuracy=score['accuracy'], macro_f1=score['macro_f1']))
        predictions.extend(dict(source_id=row['source_id'], label=row['label'], corruption_seed=corruption_seed,
                                noise_rate=rate, prediction=int(y)) for row, y in zip(records, pred, strict=True))
    write_csv(output / f'{model}-{seed}-{arm}.csv', predictions)
    print(model, seed, arm, [round(100*x['accuracy'], 3) for x in summaries], flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/06_review_checks'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    records = read_jsonl(FINAL / 'test_inputs.jsonl')
    articles = [Article(r['source_id'], r['label'], r['texts']['0.0']) for r in records]
    assert len(articles) == 7600
    original, _ = corrupt_corpus(articles, .20, 41001)
    assert original == [row['texts']['0.2'] for row in records]
    extra, corruption_stats = [], []
    for seed in (41002, 41003):
        texts, stats = corrupt_corpus(articles, .20, seed)
        extra.append((seed, .20, texts))
        corruption_stats.append(dict(seed=seed, **stats))
    print('Original swaps verified; new test views prepared.', flush=True)
    summary = []
    schedule_views = [(41001, float(rate), [r['texts'][rate] for r in records])
                      for rate in ('0.0', '0.05', '0.1', '0.2')]
    for arm in ('clean', 'augmented'):
        checkpoint = ROOT / 'results/03_cnn_development/seed_42' / arm / 'cnn_recipe.pt'
        model, vocabulary, cfg = load_cnn_variant(checkpoint)
        validation = read_jsonl(ROOT / 'results/03_cnn_development/validation_inputs.jsonl')
        saved_path = checkpoint.parent / 'cnn_recipe_validation_predictions.csv'
        with saved_path.open(newline='', encoding='utf-8') as stream:
            saved = [row for row in csv.DictReader(stream) if float(row['noise_rate']) == 0.0][:128]
        assert [(r['source_id'], r['label']) for r in validation[:128]] == [(r['source_id'], int(r['label'])) for r in saved]
        pred = predict_cnn_variant(model, [r['texts']['0.0'] for r in validation[:128]], vocabulary, cfg)
        assert np.array_equal(pred, [int(r['prediction']) for r in saved])
        summary.extend(score_views(lambda texts: predict_cnn_variant(model, texts, vocabulary, cfg),
                                   schedule_views, records, 'scheduled-word-cnn', 42, arm, args.output))
        write_json(args.output / 'progress.json', summary)
    for name in ('word-lr', 'word-cnn', 'subword-cnn'):
        for seed in (42, 43, 44):
            for arm in ('clean', 'augmented'):
                predict, path = saved_model(name, seed, arm)
                with path.open(newline='', encoding='utf-8') as stream:
                    saved = [r for r in csv.DictReader(stream) if float(r['noise_rate']) == .20][:128]
                assert [(r['source_id'], r['label']) for r in records[:128]] == [(r['source_id'], int(r['label'])) for r in saved]
                assert np.array_equal(predict(original[:128], None), [int(r['prediction']) for r in saved])
                summary.extend(score_views(lambda texts: predict(texts, None), extra, records, name, seed, arm, args.output))
                write_json(args.output / 'progress.json', summary)
    write_json(args.output / 'inference_summary.json', dict(original_swap_seed=41001, additional_views=corruption_stats,
               scheduled_checkpoint_seed=42, model_fit_seeds=[42,43,44], results=summary))
    write_csv(args.output / 'inference_summary.csv', summary)


if __name__ == '__main__':
    main()
