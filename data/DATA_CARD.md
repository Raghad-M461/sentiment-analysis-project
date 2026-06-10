# Data Card — Expanded 3-Class Sentiment Dataset

**Project:** sentiment-analysis-project
**Branch:** `feature/dataset-expansion`
**Version:** 1.0 (3-class: Positive / Negative / Neutral)
**Built by:** `scripts/make_dataset.py` (reproducible, `SEED=42`)

---

## 1. Dataset Sources

This dataset *extends* the original project dataset rather than replacing it. The
original hand-authored sentences are kept unchanged and become a documented
subset; real customer-review sentences are added on top, and a Neutral class is
introduced using real data.

### Original dataset source
- **Name:** Original manual customer-feedback set (Task 3 / preprocessing task)
- **Origin:** Authored by hand for the project, generated from a small number of
  sentence templates (e.g. *"A {adjective} {noun} overall."*, *"Honestly the
  {noun} is {adjective}."*).
- **Size:** 80 sentences (40 Positive, 40 Negative).
- **Domain:** Short B2B software/service feedback (dashboard, onboarding,
  delivery, support team, etc.).
- **Role here:** Retained in full as a labelled subset, tagged `source =
  original_manual` so it is fully traceable.

### Additional dataset source
- **Name:** Customer Review Datasets (Hu & Liu, 2004), accessed via NLTK
  `product_reviews_1` + `product_reviews_2`.
- **Authors / origin:** Minqing Hu and Bing Liu, University of Illinois at
  Chicago. Reviews were collected from Amazon.com and annotated at the sentence
  level with opinion polarity and strength.
- **Reference papers:** Hu & Liu, *"Mining and Summarizing Customer Reviews"*
  (KDD-04); Hu & Liu, *"Mining Opinion Features in Customer Reviews"* (AAAI-04).
- **Products covered:** digital cameras (Canon G3, Nikon Coolpix 4300, Canon
  PowerShot SD500, Canon S100), mobile phones (Nokia 6610, Nokia 6600), MP3
  players (Creative Zen, MicroMP3, iPod), DVD player (Apex AD2600), routers
  (Linksys, Hitachi), antivirus (Norton), and a Diaper Champ.
- **Annotation scheme used to derive labels:** each sentence's opinion tags were
  summed. Net positive → **Positive**; net negative → **Negative**; no opinion
  tag → **Neutral**. Sentences whose tags summed to exactly zero (conflicting
  opinions) were discarded as ambiguous.

### Licensing information
- **Original subset:** authored for this project; no third-party rights.
- **Hu & Liu data:** distributed by the authors for **research/academic use**
  from `http://www.cs.uic.edu/~liub/FBS/FBS.html` and bundled in the NLTK data
  distribution. It does **not** carry a formal open-source/SPDX licence (e.g. MIT
  or CC); use is governed by the authors' research-use terms and should be
  **attributed via the KDD-04 / AAAI-04 papers above**. This is acceptable for an
  internship/learning project; it is **not** cleared for unrestricted commercial
  redistribution.

---

## 2. Dataset Statistics

| Statistic | Value |
|---|---|
| Total samples | **1,500** |
| Number of classes | **3** (Positive, Negative, Neutral) |
| Samples per class | **500 / 500 / 500** (balanced) |
| Original samples retained | 80 (40 Pos, 40 Neg) |
| Real samples added | 1,420 (460 Pos, 460 Neg, 500 Neu) |
| Avg. words per sentence | ~14 (real) vs ~5 (original) |
| Reproducibility | fixed seed `42`; rebuild with `scripts/make_dataset.py` |

### Samples per class (provenance breakdown)

| Class | Original (manual) | Added (Hu & Liu real) | Total |
|---|---|---|---|
| Positive | 40 | 460 | 500 |
| Negative | 40 | 460 | 500 |
| Neutral | 0 | 500 | 500 |
| **Total** | **80** | **1,420** | **1,500** |

---

## 3. Class Distribution

The dataset is deliberately balanced at 500 samples per class to avoid
class-imbalance effects in the evaluation. The chart below (`class_distribution.png`)
shows each class split into the original manual portion and the added real
portion.

![Class distribution](class_distribution.png)

---

## 4. Preprocessing

Two preprocessing layers apply: **dataset-build cleaning** (applied once when the
CSV is created) and the **model preprocessing pipeline** (applied at train time,
unchanged from the previous task so the model stays a fixed variable).

### Dataset-build cleaning (`make_dataset.py`)
- Detokenised each annotated sentence back to readable text
  (`TreebankWordDetokenizer`) and collapsed repeated whitespace.
- **Length filter:** kept sentences of 4–40 words (drops headers, fragments, and
  run-on lines).
- **Non-text filter:** dropped lines with no alphabetic characters.
- **Deduplication:** removed case-insensitive duplicate sentences.
- **Ambiguity filter:** dropped sentences whose opinion tags summed to zero.
- **Balancing:** randomly subsampled (seed 42) to 500 per class; the 80 originals
  are placed first so they are always retained.

### Model preprocessing pipeline (`preprocessing.py`, unchanged)
- **Tokenization:** NLTK `word_tokenize`.
- **Lowercasing:** applied.
- **Negation handling:** negation marking — tokens inside a negation scope are
  prefixed with `neg_` (`not good` → `not neg_good`); scope closes at punctuation
  or contrast words (*but, however, though*).
- **Stop-word handling:** a **sentiment-aware** stop list — NLTK's English list
  **minus** polarity-bearing words (`not`, `never`, `but`, `very`, `too`, `so`,
  …), which a generic list would wrongly delete.
- **Normalisation:** lemmatization (WordNet), preserving the `neg_` prefix.
- **Decision:** this exact configuration was the best performer in the previous
  task (82.5% on the old data) and is held fixed here so any change in results is
  attributable to the **data**, not the model.

---

## 5. Known Limitations

### Remaining biases
- **Domain bias:** the added data is consumer-**product** reviews (electronics,
  software). The original was B2B **service** feedback. Neither covers social
  media, hospitality, finance, or non-English text.
- **Source bias:** all real data comes from Amazon.com reviewers (~2003–2005),
  so it reflects that era's products and writing style.

### Coverage gaps
- No domain-matched **service** data was added — the real data is products, not
  the dashboards/onboarding the original described.
- Sentences are short-to-medium; long, multi-sentiment reviews and sarcasm are
  under-represented.
- Single language (English) only.

### Label-quality concerns
- Hu & Liu labels are human annotations and contain **noise** — some sentences
  read as mislabelled (e.g. *"set up was easy and we enjoyed it for just over a
  week"* tagged Negative). This noise is real and was kept deliberately, because
  it reflects genuine annotation difficulty.
- The **Neutral** class is defined as "no opinion tag." This conflates truly
  neutral statements with factual/contextual sentences and is the fuzziest class.
- Original 80 labels are clean but trivially separable (template-generated).

### Dataset risks
- **Licence:** Hu & Liu data is research-use only — do **not** ship commercially
  without clearing rights.
- **Mixed difficulty:** the 80 synthetic originals are far easier than the real
  sentences, so they slightly inflate scores; they are a small fraction (5.3%) so
  the effect is minor.
- **Small by modern standards:** 1,500 samples is enough to expose difficulty but
  too small to train a strong general sentiment model.
