# eeg-to-image-ni26

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

My entry to **Track 1 (EEG-to-Image)** of the [Neural Interfaces 2026 challenge](https://neural-interfaces26.github.io/) (Meta FAIR Brain & AI, Alljoined): from a single EEG epoch recorded while a person looks at a natural image, predict the 1536-dimensional DINOv2-giant embedding of that image, so that the image can be retrieved from a gallery.

**Result.** Sealed final phase, top-5 retrieval = **0.000899** (best of my sealed submissions), 62nd on the final sealed leaderboard. The winning entry scored 0.001433.

## The task

The sealed evaluation runs on new participants and new images. For each participant and image, the platform averages the raw predictions over repeated presentations, L2-normalizes the average, and ranks it against the frozen embeddings of every target image of the sealed split. A query counts if the viewed image is in the top 5. Inference runs on the organizers' workers with a 2-hour limit, and each team gets one sealed submission per day.

Two things follow from this protocol and shaped everything else:

- The official validation split shares its subjects with training, so it says little about the sealed score. A first EEGNet scored about 11 times chance on it and about chance on the sealed set.
- Because raw predictions are averaged before normalization, the norm of each prediction acts as a per-trial weight. A model can learn to be "loud" on clean trials and "quiet" on noisy ones.

## The approach

| Component | What it does | Code |
|---|---|---|
| REVE-base, fine-tuned | The public REVE EEG foundation model (brain-bzh), fine-tuned end to end with an attention-pooling or flat retrieval head, CLIP-style contrastive loss against the DINOv2 embeddings | `hand/models_v2.py`, `hand/train_v3.py` |
| Free output norm | Outputs are not normalized during training (as with NeuralBench's ClipLoss, `norm_kind="y"`), so the norm becomes a per-trial confidence that the platform's averaging uses | `train_v3.py --rawnorm` |
| Trial averaging | With probability 0.5, a training trial is replaced by its mean with 1 to 6 other trials of the same image (other subjects and repetitions): the evoked response adds up, the noise averages out | `train_v3.py --mix 0.5 --mixk 6` |
| Small conv nets | Spatio-temporal convolutional encoders trained from scratch, two of them with THINGS-EEG2 as extra data; they bring diversity to the ensemble (22% of its weight), with outlier channels zeroed at inference | `hand/models_v2.py` |
| Ensemble | Weighted sum of raw outputs; REVE outputs are re-centred on the batch mean with an adaptive strength (it costs nothing on clean data and helps if the whole sealed cohort is shifted) | `hand/make_submission_v3.py` |

## Evaluation: a proxy for the sealed phase

Every decision was taken on a local proxy built to imitate the sealed protocol: 4 held-out subjects × 1,500 held-out images, predictions averaged per (subject, image), a gallery made of those 1,500 images only, and top-5 extrapolated to a larger gallery with a beta-binomial model (`hand/eval_h1500.py`). Comparisons between variants are paired and bootstrapped (`hand/apparie/`).

A first version of the proxy mixed typical query images with a large pool of never-queried ones. It rewarded a bias towards typical images and underestimated the REVE ensemble by about 30%. Its replacement, with a homogeneous gallery, was calibrated on two sealed scores (a single conv net and ZIP 12).

## Results

| Sealed submission | Content | Top-5 |
|---|---|---|
| Earlier best | A single convolutional encoder | 0.000475 |
| ZIP 12 (19/11) | 4 fine-tuned REVE-base + 4 conv nets | 0.000826 |
| ZIP 14 (20/11) | 12 REVE-base + 4 conv nets | 0.000797 |
| ZIP 15 (21/11) | 9 free-norm REVE-base (3 trained with trial averaging) + 3 earlier REVE + 4 conv nets, batch re-centring | **0.000899** |

The sealed cohort is small (about 11 subjects, about 2 repetitions per image), so differences under about 5% are noise.

## What I learned

- **Build the evaluation before the model.** Most of the gains came after the proxy matched the sealed protocol; before that, the official validation split pointed the wrong way.
- **The gap to the top is per-trial quality, not aggregation.** My aggregated score is about 1.6 times my per-trial score, against about 2 for the leading teams. This ratio grows mechanically with per-trial strength, so it is a symptom, not a lever: reaching the podium needed per-trial predictions about 20 to 28% better.
- **More models is not more signal.** Going from 4 to 12 REVE models did not help (ZIP 14). The last gain came with new training recipes (free norm, trial averaging), not with more members.
- **Engineering under a quota.** Training ran on Kaggle (2 × T4 sessions, about 30 GPU-hours a week). The last day was fully automated: generated launchers with time caps, download and health checks of the checkpoints, selection of the bets on the proxy, then building and testing the submission (`kaggle/outils/`).

## What is and is not in this repository

In the repository: the training, evaluation, packaging and Kaggle orchestration code of the final submission. Comments and logs are in French (working notes).

Not in the repository:

- The data (Alljoined-1.6M, THINGS-EEG2) and the REVE weights, which come from their own sources under their own licences.
- The checkpoints and the submission archives (1.7 GB).
- The many exploratory scripts (ridge baselines, test-time augmentation, subject alignment, gating) that did not make it into the final submission.

## Layout

```text
hand/
  models_v2.py            architectures: REVE retrieval model, conv nets
  train_v3.py             training (proxy runs with held-out subjects, or final runs on everything)
  data.py, metric.py      data plumbing to the challenge contract, top-k metric
  eval_h1500.py           sealed-like proxy (chains eval_scelle_like.py and eval_g16k.py)
  apparie/                paired bootstrap comparisons of variants
  make_submission_v3.py   packs an ensemble into a Codabench submission
  test_submission_v3.py   runs a submission locally before upload
kaggle/
  runner.py, train_sat_b/ a Kaggle training launcher and its metadata
  things_prep.py          THINGS-EEG2 windows in the challenge's channel layout
  outils/                 last-day automation: launcher generation, fetch, selection, packaging
```

## Running it

The code expects to sit at the root of a clone of the official [competition repository](https://github.com/neural-interfaces26/2026-competition), whose `benchmark_utils` it imports and whose data cache it reads (`tracks/image_decoding/data/`). Training a REVE model needs the public REVE-base weights (environment variable `NI_REVE`) and a GPU.

```bash
# proxy run: 4 held-out subjects
python hand/train_v3.py --name reve_mix --arch reve --rawnorm --mix 0.5 --mixk 6 --holdout 4
# sealed-like evaluation of one or more runs
python hand/eval_h1500.py reve_mix
# package and test a submission
python hand/make_submission_v3.py ...
python hand/test_submission_v3.py submissions/<name>
```

## License

MIT for my code (see `LICENSE`). The challenge data, the REVE weights and the organizers' code keep their own licences.
