# Archived: the primary-assessment (v1) Streamlit workflow

These files are no longer part of the app. They are kept only for reference.

- **What was here.** The upload → preprocess → zero-shot teacher → train → evaluate → infer pages (`src/ui/page1…page8`)
  and their helpers:
  - `src/inference/context.py` and `src/inference/pipeline.py`;
  - `src/visualization/`;
  - `scripts/validate_local_datasets.py`;
  - their tests.
- **What replaced it.** The app (`app.py`) now runs only on the final pipeline:
  - `src/app/core.py`;
  - `src/ui/{overview, data_pipeline, teacher, student, evaluation, inference, results, metrics_guide}.py`.
- **Running these files.** They will not import from here as they are; to run them, move them back to their original paths.
