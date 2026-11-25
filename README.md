# IKT466 Overleaf Report Package

To compile this report in Overleaf, upload `IKT466_Overleaf_Revised.zip` via **New Project → Upload Project**, set `main.tex` as the main document, and compile using **pdfLaTeX** (with BibTeX handled automatically). The archive is self-contained and bundles all chapters, appendices, tables, figures, bibliography entries, evidence extracts, and University of Agder cover assets required for typesetting.

## Project Scope
This report investigates four-class topic classification on AG News under reproducible, severity-controlled transpositions of adjacent interior letters. Within a fixed budget of 20,000 labeled training articles, we compare a word and bigram TF-IDF logistic regression baseline (**Word TF-IDF LR**) against a word-level convolutional neural network (**Word CNN**) and a character-fragment-enriched convolutional network (**Subword CNN**), alongside a schedule-matched control (**Scheduled Word CNN**). Two focused investigations evaluate (1) classification sensitivity across progressive spelling-swap targets (0%, 5%, 10%, and 20%) and (2) the effect of static training-time swap augmentation (replacing 50% of training articles per class at the 10% swap target).

## Repository Layout
- `main.tex`: Master LaTeX document.
- `config/`: Package preamble (`preamble.tex`) and report metadata (`metadata.tex`).
- `frontmatter/`: University of Agder title page (`titlepage.tex`) and abstract (`abstract.tex`).
- `chapters/`: Chapters 1–7 covering Introduction, Background, Methodology, Experimental Design, Results, Discussion and Limitations, and Conclusion.
- `appendices/`: Detailed model configurations, seed-level and follow-up tables (`a_decision_register.tex`, `b_full_results.tex`), reproducibility instructions (`c_reproducibility.tex`), and AI assistance disclosure (`d_ai_statement.tex`).
- `figures/`, `tables/`, `data/`: Precomputed PDF/PNG figures, LaTeX table fragments, and verified JSON/CSV experimental outputs.
- `project_requirements.pdf`: Official five-page IKT466 course project specification.
- `template/`: Attributed University of Agder template source and cover assets (see `TEMPLATE_PROVENANCE.md`).

## Reproducing Figures and Follow-up Checks
In the full experimental repository, running `python main.py figures` regenerates selected tables and figures from saved measurements without retraining. Overleaf uses the precomputed figures and tables included in this archive. Note that the full code archive, dependency lockfile, split definitions, saved predictions, and neural checkpoints must be submitted alongside the report per the course requirements.

To reproduce the post-freeze follow-up evaluations (scoring the saved seed-42 Scheduled Word CNN checkpoints across all test targets and evaluating the three-seed core models on the two extra 20% swap draws) from the full project root, run:

```powershell
.\.venv-ml\Scripts\python.exe report/data/evaluate_review_followup.py --output results/review_replay_01
.\.venv-ml\Scripts\python.exe report/data/analyze_review_evidence.py --output results/review_replay_01/saved_analysis.json
```

The resulting summaries are bundled in `data/review_inference_summary.json` and `data/review_saved_analysis.json`.
