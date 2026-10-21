# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall accuracy on the held-out set

Accuracy on the held-out test split is at least 0.62 on all three seeds.

**Why this target:**
My three labels are not even, so a model that always guessed `hot_take` would score about 0.47 on my file. A target of 0.62 means the model has to be clearly better than that one trick, while leaving room for the fact that my hardest boundary is a judgement call I sometimes made slowly myself. I would not trust a sorter below this on a forum where half the posts argue with each other.

---

## 2. The weakest label still works

Every label has an F1 of at least 0.45 on all three seeds, and `analysis` has an F1 of at least 0.60.

**Why this target:**
`banter` is only 42 of my 225 rows, so about 6 land in the test split and one misread post swings its F1 by a lot. That is why its floor is low. `analysis` gets the higher bar because it has 77 examples and it is the label the whole project is for. A tool that cannot find the posts with facts in them has no reason to exist.

---

## 3. Not just guessing the biggest label

The model predicts each of the three labels at least 5 times across the held-out set, and no single label is predicted for more than 70% of the test posts.

**Why this target:**
`hot_take` is 47% of my data, and the failure I most expect is a model that learns to say `hot_take` and nothing else. Accuracy alone would hide that, since guessing it every time already scores near 0.47. The 70% ceiling matches the cap the brief puts on the data itself, so the model is held to the same standard as my labelling.

---

## 4. Agreement with a second labeller

On the 20 posts labelled cold, my labels and a second person's labels agree on at least 14 of 20, which is 70%.

**Why this target:**
If two people cannot apply my definitions the same way, then the model was trained on noise and the other four criteria are measuring my inconsistency rather than its learning. I picked 70% rather than 90% because my hardest boundary is real. Deciding whether a post that names one driver is analysis or a hot take took me several seconds on maybe a tenth of the file.

---

## 5. Confident predictions are the right ones more often

Predictions where the model's top probability is at least 0.80 are correct at least 15 percentage points more often than predictions below 0.60.

**Why this target:**
A classifier that is equally wrong when sure and unsure is unusable for triage, which is the only thing I would ever want this for. Sorting a day of comments means reading the confident `analysis` ones first and ignoring the rest. I chose a gap in percentage points rather than two separate accuracy numbers, because the gap is what makes the confidence score worth reading.

---
