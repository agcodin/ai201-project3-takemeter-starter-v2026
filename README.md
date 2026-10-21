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

**How I labelled.** All 225 rows were labelled against the definitions above, reading each post. The labels were produced with Claude working from my definitions, and I have disclosed that in How I Used AI below rather than describing it as hand labelling.

**Counts.**

| Label | Count | Share |
|---|---|---|
| hot_take | 106 | 47% |
| analysis | 77 | 34% |
| banter | 42 | 19% |
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

**What this means for next unit.** The 20 cold rows the brief asks for are not in this file yet. Those have to be labelled by me with no help, and the agreement report in unit 6 depends on them being genuinely independent, so pre-labelled rows cannot stand in for them.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



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
