# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

TakeMeter sorts comments from the RaceFans Formula 1 community into three kinds of post. RaceFans is a public F1 news site whose article comment sections run long and argumentative, often several dozen replies deep on a single race report. Reading them, the split that kept appearing was not good against bad, it was whether a commenter brought a checkable detail, asserted an opinion without one, or was joking. The classifier sorts posts into `analysis`, `hot_take` and `banter`, which would let someone skim a day of comments by reading the argued ones first.

The three distinctions I wrote down after reading about 40 comments were these. Some posts name a number, a season or a rule, and most do not. Some posts are replies to other commenters rather than to the article. A noticeable share are jokes that take no position at all.

---

## Label Taxonomy

### `analysis`

Supports a claim about Formula 1 with a specific checkable detail such as a number, a date, a season, a race or a named rule.

> To think in 1906 the circuit length at Le Mans GP was 65 miles and race distance was 800 miles.

> @bernasaurus The same for long-time fans since the difference is only 2 or 3 laps, depending on the circuit, i.e., no different from any race that loses 2 or 3 laps for as many extra formation laps.

### `hot_take`

States an opinion or prediction about Formula 1 confidently, with no checkable detail offered in support.

> Because F1's remote TV production works so well… *rolls eyes* Ever since F1 has shifted production away from trackside the product has been slowly getting worse.

> Early or late, no coronation is relevant with these joke cars. For Antonelli's career's sake, I hope he one day gets to win a race with a proper F1 car.

### `banter`

Jokes, sarcasm, or talk about the comment section itself rather than an argument about the sport.

> Yes, these modern kids of today with their fast cars! In my day the cars were slow, the races were long, the drivers were old, and we liked it!

> Maybe they should reduce the Monaco GP to zero laps and just score based on qualifying, you know, to save a bit of time for everyone involved.

### The hardest boundary

`analysis` against `hot_take`, and it is not close. Half of this community writes in a register that sounds authoritative whether or not anything is being cited.

**The decision rule.** If the post names a specific checkable detail and uses it to support what it is claiming, it is `analysis`, however heated the tone. A detail mentioned in passing does not count, and neither does a confident number with nothing behind it. "Ferrari spent more than Mercedes for years and won nothing" is `analysis`, because the spending claim is the support. "These cars are a joke" is `hot_take` even when the writer clearly knows the sport.

The rule I had to add partway through covers sarcasm that contains a fact. A post can be a joke and still argue something, and I kept those as the argued label, with `banter` reserved for posts that take no position at all.

---

## The Dataset

**Where it came from.** The public comment sections of RaceFans articles published in September and October 2026, collected with a short script over about 25 articles. Reddit was the first choice, since the brief suggests it, but reddit.com refuses automated requests and its JSON endpoints returned HTML, so I moved to a public site that does not. The comments are whole and untrimmed, between 70 and 700 characters.

**How I labelled.** All 225 rows were labelled against the definitions above, reading each post. The labels were produced with Claude working from my definitions, and I have disclosed that in How I Used AI below rather than describing it as hand labelling. I then labelled 20 of those rows cold, working from the definitions alone with no tools and without looking at what the pre-labelling had assigned, and those labels replaced the pre-labels on those rows. They are marked `cold` in the note column, and where my label differed the note records what the pre-label had been.

**What the cold 20 found.** My cold labels matched the pre-labels on 12 of 20, which is 60%. Criterion 4 asks for 70% agreement with a second labeller, and the pre-labelling is close enough to a second labeller that 60% is worth reporting now rather than discovering it next unit.

The eight disagreements are not spread evenly. Six of them involve a post that is partly a joke and partly a claim, which is the seam between `banter` and the other two labels. One example is "Wow, clearly Alonso is now happy just to remain in F1… My prediction: Aston Martin will be super-competitive the season after Alonso leaves them!", pre-labelled `banter` for the sarcasm and labelled `analysis` cold. My written decision rule covers `analysis` against `hot_take` in detail and says almost nothing about where `banter` stops, which is the gap. Two others went the opposite way, where a detail mentioned in passing was read as support on one pass and not the other.

The counts below include the corrected rows, and the model was retrained on the corrected file.

**Counts.**

| Label | Count | Share |
|---|---|---|
| hot_take | 108 | 48% |
| analysis | 77 | 34% |
| banter | 40 | 18% |
| **Total** | **225** | |

No label is above the 70% cap. `banter` being thin is the known weakness, and criterion 2 was written with that in mind before anything was trained.

**Three hard cases.**

1. *"Pretty disgraceful all Circuits have put up prices for 2027 but most grand Prix will be 3 to 4 laps shorter."* This could be `hot_take` on tone alone, since "disgraceful" is the loudest word in it. It names a price rise and a lap count, and both are checkable, so the decision rule makes it `analysis`.

2. *"We have 4 teams who can win and get pole. That's pretty good in my view."* The first sentence is checkable and the second announces it as an opinion. I kept it `hot_take`, because the number is the claim rather than support for one. This is the case I am least sure about.

3. *"@BenHur, Very nice work. Max at the top once again, makes sense he's more race fast than qualy fast."* It opens as comment-section chatter and then makes a claim about a driver. I kept it `hot_take`, since the claim is the substance and the greeting is packaging. Posts that are only packaging are `banter`.

---

## The Training Run

**Starting model:** `distilbert-base-uncased`, the notebook's default.

**Settings:** three epochs, learning rate 2e-5, batch size 16, 128 tokens per post, seed 42, with a 70/15/15 split. None of them changed. My comments average 263 characters, which is comfortably inside 128 tokens for nearly all of them, so the one setting I might have raised had no reason to move.

**Device:** `mps`, the Apple Silicon GPU, with torch 2.14.1. The run took under a minute. `results.json` records this, and it matters because two people running this notebook on the same data and seed can get different numbers on different hardware.

The run wrote `results.json` and `test_split.csv`, both committed. Reading the numbers closely is next unit's work, and my criteria were filed two commits before the model existed.

---

## How I Used AI

**Moment 1.** I used Claude to collect the posts. Reddit blocks automated requests, so it wrote a script that pulls comment text out of RaceFans article pages, which returned 366 comments before deduplication and length filtering. What I changed was the extraction, since the first version matched the wrong div and returned nothing on every article. Reading the page source showed the text lives in `comment-entry` rather than `comment-content`.

**Moment 2, and the pre-labelling disclosure.** Claude applied my three definitions to all 225 posts, and the `note` column records that for every row. This is the pre-labelling the brief permits, used on the whole file rather than a batch. Where a post sat on the boundary, the note says so and names the two labels it could have taken, which is how the three hard cases above were found. The decision rule was written before the labelling started, which is what kept the borderline calls consistent rather than drifting.

**Moment 4, unit 6.** I had Claude read the confusion matrix against my cold-20 disagreement notes and say whether the `banter` collapse looked like thin data or inconsistent labelling. It said both, and pointed at the overlap, which is that six of my eight cold disagreements sat on the same seam the matrix collapses. I used that to pick adding examples over relabelling, and the result was that adding examples did nothing, which suggests the relabelling half of that answer mattered more.

**What the AI did not touch.** The 20 cold rows are mine. I labelled them with the definitions in front of me, no tools and no sight of what the pre-labelling had put there, which is what makes the 60% agreement figure above worth anything. Comparing those 20 against the pre-labels is how the gap around `banter` surfaced.

# Unit 6

## Baseline vs. Trained

**My prediction, written before running the comparison:** the trained model will beat the zero-shot baseline on `hot_take`, because "a confident claim with no support offered" describes an absence and gives a zero-shot model nothing to match on, while 108 examples give a trained model the house style to learn.

`baseline.py` ran `facebook/bart-large-mnli` over the same 39 held-out posts the trained model was scored on, using only my three label definitions as the prompt. The trained column is seed 42 from the improvement run.

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.333 | 0.615 | +0.282 |
| Macro F1 | 0.221 | 0.495 | +0.273 |
| F1 — `analysis` | 0.511 | 0.818 | +0.308 |
| F1 — `banter` | 0.154 | 0.000 | -0.154 |
| F1 — `hot_take` | 0.000 | 0.667 | +0.667 |

The prediction held. The baseline never once guessed `hot_take` across 39 posts, exactly as expected, and that single label is where most of the +0.28 accuracy gap comes from.

The reverse is the more interesting half. The baseline is the only one of the two models that ever identifies `banter`, scoring 0.154 against my trained model's 0.000. A general model reading the word "banter" and the phrase "jokes, sarcasm" can spot a joke, while mine never predicts the label at all. Fine-tuning on 75 examples made that label worse than not training at all.

**What my definitions were worth, measured.** I ran the baseline twice, once with `label_definitions.txt` and once without it, where it falls back to bare label names.

| Baseline prompt | Accuracy | Macro F1 |
|---|---|---|
| My three written definitions | 0.333 | 0.221 |
| Bare label names only | 0.410 | 0.272 |

My definitions made the zero-shot model **worse**, by 7.7 points of accuracy. The bare-names run gets `hot_take` partly right (F1 0.549) and scores zero on `analysis`, which is the exact mirror of what the definitions do. I take this as evidence that my definition of `analysis` ("supports a claim with a specific checkable detail") reads to an entailment model as a description of almost any substantive post, so it swallows everything. The figure I report as my baseline is the 0.333 definitions run, since that is the prompt my project actually specifies.

## Run Log — Before

Three seeds from section 6, written to `results_three_seeds_before.json`, on `mps` with torch 2.14.1. The dataset at this point was the 225 rows submitted in unit 5.

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.62 on all seeds | 0.765 | 0.676 | 0.618 | MISSED |
| 2. Every label F1 ≥ 0.45, `analysis` ≥ 0.60 | both, all seeds | banter 0.000 / analysis 0.909 | banter 0.000 / analysis 0.737 | banter 0.000 / analysis 0.588 | MISSED |
| 3. Each label predicted ≥ 5 times, none above 70% of predictions | both, all seeds | banter 0, hot_take 71% | banter 0, hot_take 79% | banter 0, hot_take 85% | MISSED |
| 4. Cold-20 agreement with a second labeller | ≥ 14 of 20 | 12 of 20 | 12 of 20 | 12 of 20 | MISSED |
| 5. Confident predictions beat unsure ones by ≥ 15 points | ≥ 15 pts | 0.0 | 0.0 | 9.1 | MISSED, and unmeasurable as written |

Criterion 4 is a property of the labels rather than of a split, so the same figure goes in all three columns.

**Spread across seeds.** Accuracy moved 0.147 between the best and worst seed, and `analysis` F1 moved 0.321. The notebook flags anything above about ten points, and it is right to. With 34 test posts, two or three posts landing differently moves accuracy by six points on their own, so these three numbers describe my splits as much as my model.

**Confusion matrix**, seed 42, before the improvement:

| true \ predicted | analysis | hot_take | banter |
|---|---|---|---|
| **analysis** | 10 | 2 | 0 |
| **hot_take** | 0 | 16 | 0 |
| **banter** | 0 | 6 | 0 |

Six real `banter` posts were called `hot_take`, and nothing went the other way, because the model never predicts `banter` at all. The entire `banter` row collapses into one column.

## Verdicts and Diagnoses

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | ≥ 0.62, all seeds | MISSED | 0.618 on seed 2024, two thousandths under. The target has to hold across trials, and a miss by 0.002 is still a miss. Reading it as met would be the generous reading the brief warns about. |
| 2 | Every label F1 ≥ 0.45, `analysis` ≥ 0.60 | both | MISSED | `banter` is 0.000 on all three seeds, so this fails before `analysis` is even considered. `analysis` also fell to 0.588 on seed 2024. |
| 3 | Each label predicted ≥ 5 times, none above 70% | both | MISSED | `banter` was predicted 0 times on every seed, and `hot_take` took 71%, 79% and 85% of predictions. Both halves fail, and the trend across seeds is in the wrong direction. |
| 4 | Cold-20 agreement | ≥ 14 of 20 | MISSED | 12 of 20, measured in unit 5 before any of this. |
| 5 | Confident beat unsure by ≥ 15 points | ≥ 15 pts | MISSED | 0.0, 0.0 and 9.1 points. See the revision below, because the number I can compute is not the number the criterion names. |

**Diagnosis for criteria 2 and 3, which are the same failure.** `banter` had 40 examples out of 225, so roughly 28 in training and 6 in test. The model learns that predicting `hot_take` on every borderline post is the cheapest way to cut loss, and with `hot_take` at 48% of the data that strategy is nearly as accurate as learning the boundary. The confusion matrix shows every `banter` post going to `hot_take` and not one going the other way, which is a direction rather than an error rate.

My agreement report is what separates thin data from bad labels here, and it points at both. Six of the eight cold-20 disagreements were posts that are part joke and part claim, which is exactly the `banter` against `hot_take` seam the matrix collapses. So the model is failing on a boundary I did not apply consistently myself, on the label I gave it the fewest examples of.

**Diagnosis for criterion 1.** Accuracy is carried by `hot_take`, which the model gets nearly free, and by `analysis`, which it genuinely learns. Seed 2024 fell under the line because its test split drew more `banter`, and every one of those is an automatic miss. This is the same failure as criteria 2 and 3 rather than a separate one.

**Criterion 5, revised because it could not be measured.**

> **Original:** Predictions where the model's top probability is at least 0.80 are correct at least 15 percentage points more often than predictions below 0.60.
>
> **Revised in unit 6:** Predictions in the most confident third are correct at least 15 percentage points more often than predictions in the least confident third.
>
> **Why revised:** No prediction my model makes ever reaches 0.80. I checked the raw softmax output on seed 42, and the highest top-probability across all 34 test posts is 0.574, with a mean of 0.499. Both bands my criterion names are empty, so it has no value at all rather than a bad one. Three labels and three epochs on 157 training rows leave the model barely off uniform, which I did not anticipate when I wrote the number. The revision measures the same idea against thirds, which is what `results.json` already records. It still misses, at 0.0, 0.0 and 9.1 points.

## Agreement Report

Pending. `data/staff_labels.csv` has to come from the TF and I do not have it yet. My 30 labels under the staff taxonomy are in `my_staff_labels.csv`, written before seeing the key, and `agreement.py` has not been run.

The cold-20 check from unit 5 stands as the evidence used in the diagnoses above, at 12 of 20, or 60%.

## The Improvement

**What I changed:** added 35 `banter` examples, taking the label from 40 rows to 75 and the file from 225 rows to 260. They come from the same RaceFans comment sections, labelled under the same definitions, and are marked in the `note` column. Nothing else changed, and the taxonomy and criteria are untouched.

**Why:** the diagnosis named too few `banter` examples as the cause of criteria 2 and 3 failing, and the matrix showed the whole label collapsing into `hot_take`. Of the three options in the brief, adding examples to the thinnest label is the one that addresses that directly. Relabelling the boundary was the alternative, and I rejected it because the boundary disagreements are genuine ambiguity rather than me applying a written rule two ways, which my cold-20 notes record case by case.

### Run Log — After

`results_three_seeds_after.json`, same three seeds, same device.

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.62 on all seeds | 0.615 | 0.590 | 0.590 | MISSED |
| 2. Every label F1 ≥ 0.45, `analysis` ≥ 0.60 | both, all seeds | banter 0.000 / analysis 0.818 | banter 0.000 / analysis 0.762 | banter 0.000 / analysis 0.737 | MISSED |
| 3. Each label predicted ≥ 5 times, none above 70% | both, all seeds | banter 0 | banter 0 | banter 0 | MISSED |
| 4. Cold-20 agreement | ≥ 14 of 20 | 12 of 20 | 12 of 20 | 12 of 20 | MISSED |
| 5. Confident beat unsure by ≥ 15 points (revised) | ≥ 15 pts | see note | see note | see note | MISSED |

**Confusion matrix**, seed 42, after:

| true \ predicted | analysis | hot_take | banter |
|---|---|---|---|
| **analysis** | 9 | 3 | 0 |
| **hot_take** | 1 | 15 | 0 |
| **banter** | 0 | 11 | 0 |

**Did it help? No, and it is worth more than if it had.**

| Measure | Before (mean of 3 seeds) | After (mean of 3 seeds) | Change |
|---|---|---|---|
| Accuracy | 0.686 | 0.598 | −0.088 |
| Macro F1 | 0.499 | 0.478 | −0.021 |
| Accuracy spread across seeds | 0.147 | 0.026 | −0.121 |
| F1 — `banter` | 0.000 | 0.000 | no change |

Accuracy went down on every seed, `banter` is still zero, and no criterion moved from missed to met. Nearly doubling a label's examples bought nothing on that label.

One thing did improve, and it is not nothing. The spread across seeds fell from 0.147 to 0.026, so the model is now reliably mediocre instead of unpredictably better. The before numbers were partly a draw, and seed 42's 0.765 was the friendliest draw in the set. Adding 35 posts made the measurement honest, which lowered the number.

The reason accuracy fell is visible in the matrix. `banter` went from 6 test posts to 11, and all 11 are still wrong, so the same behaviour now costs more. The model did not get worse. The test got harder, and more of it is now made of the thing my model cannot do.

## What's Still Broken

**Criteria 2 and 3, the `banter` label.** Still F1 0.000 and still never predicted, before and after. My next move would not be more data, because 35 more rows changed nothing. It would be to test whether the label is learnable at all at this size, by training a two-label model on `analysis` against everything else. If that scores well, the problem is three-way separation on 260 rows. If it does not, the problem is my definition. I stopped because the unit allows one improvement and I spent it on the data.

**Criterion 1, accuracy.** It is now further from the target than it was. I would not chase it directly, since accuracy here is mostly a measure of how much `hot_take` is in the test split.

**Criterion 4, agreement.** 60% against my own 70% target, and the staff check is still pending. This is the one I would fix first, because the diagnoses above all lean on it and it is the only number that distinguishes my labelling from my model.

**The gap between what I meant to capture and what the model learned.** I meant three kinds of post, separated by whether a claim is supported and whether the writer is joking. The matrix says the model learned two, which are "has specific details in it" and "everything else", and it files every joke under the second. The `banter` row of the confusion matrix is entirely in the `hot_take` column in both runs, which means the distinction I care most about, between arguing badly and not arguing at all, is the one that never reached the model.

<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
