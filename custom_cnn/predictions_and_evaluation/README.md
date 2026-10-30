# Final evaluation and predictions

Uses the epoch-27 `best.pth` from Experiment A. It must reproduce 96.32%
validation accuracy before predicting the 500 unlabelled test images.
The model runs in evaluation mode with deterministic resize, tensor conversion
and ImageNet normalization. No weights are updated.

From the repository root:

```bash
python -m custom_cnn.predictions_and_evaluation.predict --data-dir ../datasets --device cpu
python -m custom_cnn.predictions_and_evaluation.analyze --data-dir ../datasets
```

Dependencies are in `custom_cnn/requirements.txt`. `--checkpoint`, `--output-dir`,
`--batch-size` and `--num-workers` can be specified. An existing submission is
preserved; use a new output directory to repeat inference.

On the NTU cluster, create the log directory and submit the CPU batch job:

```bash
mkdir -p custom_cnn/predictions_and_evaluation/outputs
sbatch custom_cnn/predictions_and_evaluation/run.sbatch
```

Results are saved under `outputs/`. The root `.gitignore` allows final report
figures, metrics, the classification report and `submission.csv` to be committed.
Detailed prediction CSVs, provenance and Slurm logs are retained locally and ignored.

| File | Contents |
| --- | --- |
| `submission.csv` | Exactly `id,label`, 500 rows, numeric ID order; cat=0, dog=1 |
| `validation_metrics.json` | Loss, accuracy, correct/error counts, macro and per-class metrics |
| `classification_report.csv` | Per-class and macro precision, recall, F1 and support |
| `confusion_matrix.png` | Actual labels on rows, predictions on columns |
| `validation_prediction_details.csv` | All labelled validation predictions with confidence |
| `validation_examples.png` | Lowest/highest confidence examples per actual/predicted group |
| `test_prediction_details.csv` | Test filenames, IDs, predicted labels and confidence |
| `prediction_examples.png` | Test predictions spanning each class's confidence range |
| `provenance.json` | Checkpoint/submission hashes, versions, transforms and integrity checks |

IDs come from image filenames, never DataLoader positions. The required ID set is
1–500. The supplied brief's exact `id,label` schema is enforced; an actual sample
CSV can additionally be checked using `--sample-submission path/to/sampleSubmission.csv`.

Confidence is an uncalibrated softmax score. Test ground truth is unavailable,
so test accuracy and correctness cannot be reported. The example figures show
observations only; they do not infer why the model made a particular decision.
