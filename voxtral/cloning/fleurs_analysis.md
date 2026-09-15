# Analysis: what these 39 voices are worth for dataset generation

Written by Claude Code against the tables above. Every number in this section comes from `stats.md`, `failures.md`, `similarity.md` or `distinct.md`, all of which are regenerated from `ledger.jsonl` by `fleurs_report.sh`. Nothing here is from memory: if a number disagrees with a table, the table is right.

## What was measured

Every one of the 39 voices (20 shipped, 18 hybrids, 1 sparse mix) read all 332 unique sentences of the French FLEURS test split, and every take was transcribed by the same whisper-large-v3-turbo deployment. That is 12948 takes and 50.8 hours of generated audio, scored sentence by sentence, with WER and CER computed after the same normalisation on both sides.

The comparison is fair in the way that matters: all voices read the same text with the same noise seed, so a difference between two rows is a difference between two voices, not between two corpora.

Retry rounds were run afterwards, up to three attempts per degenerate take with a different noise seed each time. **Every table above still quotes the first attempt**, because reporting the surviving attempt would be best-of-N and would flatter exactly the voices that need watching. What the retries found has its own section below.

## The headline: 39 voices are 20 speakers

This is the result that decides the dataset question, and it is not close.

Every single derived voice sits between 0.84 and 0.98 cosine similarity to the shipped voice it borrowed its acoustic codes from, and 19 of 19 have that donor as their nearest neighbour out of 38 candidates. Meanwhile the 20 shipped voices sit at mean 0.22 and **max 0.58** against each other, and a hybrid sits at mean 0.26 against its *semantic* donor, which is the same as background.

The pair list in `similarity.md` has a hole in it: below 0.84 the next pair down is 0.58, and there is nothing in between. Any threshold in that empty band gives the same answer, which is why `distinct.md` uses 0.70 rather than the metric's nominal 0.50 operating point. At 0.70 the 39 voices collapse into exactly **20 clusters, one per shipped voice**, and the greedy max-min walk and the connected components agree on that number.

So A6's finding is confirmed at full scale: speaker identity lives entirely in the 36 acoustic codes. The hybrid recipe is a way to change *how* a voice reads, never *who* it sounds like. There is no acoustic diversity in this pool beyond the 20 voices Mistral shipped, and generating a dataset with `es_male` and `h_frm_esm` in it produces a dataset with one Spanish-accented male speaker in it, not two.

## What the hybrids do buy, and what they cost

They are not worthless, they are just not new speakers. Two things are real:

**They read the typical sentence better.** With degenerate takes removed, hybrids sit at median 4.2 pct and mean 10.2 pct WER against the bases' 5.3 pct and 9.0 pct. 12 of the 19 derived voices beat their own donor on the median sentence, and the gains on the foreign-accented bases are large: `h_frm_esm` reads at 1.8 pct where `es_male` reads at 9.1 pct, `h_frm_itm` at 2.4 pct where `it_male` reads at 6.9 pct, `h_frm_nlf` at 2.1 pct where `nl_female` reads at 6.6 pct. Putting a French semantic column behind a foreign timbre does exactly what it was built to do: it keeps the voice and removes the accent's damage to the words.

**They speak at a different rate.** `h_frm_esm` reads at 2.76 words per second against `es_male` at 2.18, `h_frf_caf` at 2.15 against `casual_female` at 1.82. The embedding cannot see this, and for a dataset it is a real axis of variation even when the speaker is the same.

The cost is degeneration, and it is entirely one-sided. **The 20 shipped voices produced 1 runaway take in 6640. The 18 hybrids produced 142 in 5976**, a 2.4 pct rate, and 15 of the 19 derived voices degenerate more often than their donor. A runaway is not a bad sentence, it is the model starting correctly and then looping nonsense syllables until it hits the 327.68 s token ceiling, which 65 takes did. This is where the 1614 pct and 1382 pct maxima in the WER table come from, and it is why the hybrids' mean WER (16.1 pct) is so much worse than their median (4.5 pct) while the bases' mean and median stay close.

Mean WER is therefore the wrong summary for a hybrid. Read the median for what it does normally, the runaway column for how often it fails, and ignore the mean, which is just those two quantities blended into an uninterpretable number.

## Degeneration is mostly the text, partly the recipe, never the speaker

Grouped by the semantic donor, the split is clean: the 8 voices built on `fr_female` degenerate 2.8 pct of the time, the 8 built on `fr_male` 2.6 pct, and the 3 built on `de_female` **0.0 pct in 996 takes**. The `h_def_*` family is the only derived family that never ran away, and it pays for that with the worst medians in the pool (`h_def_him` at 9.9 pct, the worst of all 39 voices). Three voices is a small sample and this should not be taken further than it goes, but the semantic column is clearly what controls stability, not the acoustic one.

On the text side, the corpus is a bigger lever than the voice list:

| sentences with | median WER | runaway |
|---|---|---|
| non-breaking / thin space | 12.2% | 5.1% |
| no non-breaking / thin space | 3.7% | 0.2% |
| guillemets | 12.2% | 6.6% |
| no guillemets | 4.4% | 0.6% |
| digits | 10.0% | 1.5% |
| no digits | 4.0% | 1.0% |

A sentence containing a non-breaking or thin space degenerates **25 times more often** than one without, and reads three times worse when it does not degenerate. French typography puts a thin space inside every guillemet pair, so these two columns cannot be separated observationally: the `probe` command exists to settle it causally by generating the same sentences with and without each, and that run has not been done yet.

Length matters less than the typography but it is not nothing: 0.4 pct runaway under 20 words against 3.4 pct over 40. Note that this contradicts A6's conclusion that length was irrelevant, which was drawn from 12 easy sentences and did not survive contact with 332.

The failures also follow the sentence more than the voice. `f1877` produced 9 runaways across different voices, `f1857` 7, `f1843` 5. Those are properties of the text, and no choice of voice fixes them.

## What the retries answered

145 takes were retried. **108 converged** (90 on the second seed, 18 on the third) and **37 ran out of budget still degenerate**. So degeneration is mostly a property of the sampled noise, not of the (voice, sentence) pair, and the right response to it is a retry loop rather than a shorter voice list.

The effect on the fragile voices is large. Quoting the best attempt instead of the first would move `h_frf_cam` from 34.0 pct mean WER to 19.4 pct and its maximum from 773 pct to 248 pct, `h_frf_nef` from 24.4 pct to 16.4 pct with its maximum falling from 1614 pct to 234 pct, and `h_frm_nlf` from 16.5 pct to 8.4 pct. Medians barely move, which is the expected signature: retries touch the 1 pct of takes that broke and nothing else. 16 of the 39 voices change at all.

**The 37 takes that never converged are the interesting ones, and they identify the culprit.** They land on only 15 distinct sentences, and the comparison against the corpus they came from is not subtle:

| feature | stuck sentences | whole corpus |
|---|---|---|
| nbsp / thin space | 15/15 (100%) | 61/332 (18%) |
| guillemets «» | 11/15 (73%) | 26/332 (8%) |
| parentheses | 5/15 (33%) | 36/332 (11%) |
| digits | 4/15 (27%) | 70/332 (21%) |
| several sentences | 4/15 (27%) | 64/332 (19%) |

Every single sentence that survived three different seeds still broken contains a non-breaking or thin space, against 18 pct of the corpus. Guillemets are present in 73 pct of them, but every guillemet pair in French typography contains a thin space, so the guillemets are most likely along for the ride. Digits and multi-sentence rows sit at their base rate and are not implicated at all.

## The probe: it is the space, and the guillemets are innocent

The causal test was then run: 8 of those sentences read by 6 voices at a pinned seed, four times each, changing one thing at a time. 192 takes.

| variant | median WER | mean WER | unusable takes |
|---|---|---|---|
| `raw`, as FLEURS ships it | 47.7% | 90.5% | 16 of 48 |
| `nbsp`, spaces normalised, guillemets kept | **10.5%** | **25.9%** | **3 of 48** |
| `quotes`, guillemets removed, spaces kept | 57.6% | 76.4% | 15 of 48 |
| `clean`, both | 18.2% | 29.6% | 4 of 48 |

Replacing the non-breaking and thin spaces with ordinary ones cuts the median WER by a factor of 4.5 and takes the unusable takes from 16 to 3. Removing the guillemets and leaving the spaces alone does nothing at all: the median moves the wrong way and 15 of 48 takes are still unusable. Doing both is no better than fixing the spaces alone, so the guillemets contribute nothing in either direction and were only ever implicated because French typography puts a thin space inside every pair.

The median take also got shorter, 18.2 s to 13.8 s, which is the same fact seen from the waveform: the raw takes were running long because they were degenerating.

**This is a one-character preprocessing fix and it is the single largest quality lever found in the whole track.** Any text going to this model should have U+00A0, U+2009 and U+202F replaced with an ordinary space first. It is worth more than the choice of voice: it takes the affected sentences from unusable to ordinary, and 18 pct of an arbitrary French corpus is affected.

Caveat on the size of the test: 48 takes per variant, 8 sentences, 6 voices. The direction is consistent and large, but the exact multiplier is not.

## The recommendation

**Use the 20 shipped voices as the diversity backbone.** They are the entire acoustic space available, every one of them has a 0.0 pct runaway rate (`casual_male` has a single one in 332), and their medians run from 0.0 pct to 9.1 pct WER. `distinct.md` lists them in most-different-first order, so if the dataset needs fewer than 20 speakers, take the first N rows of that table and they will be the most spread-out N available.

Weight `fr_female` and `fr_male` heavily. They are not just the best of the 20, they are in a different class: 0.0 pct median and 3.7 pct mean WER against a pool median of 5.3 pct, with the fastest and second-fastest delivery. Everything else is a non-native speaker reading French, which is a genuine acoustic axis for an ASR dataset but should not dominate it.

**Add hybrids only for delivery variety, and only with the retry loop on.** One rule keeps the list honest: beat your donor's median, and degenerate less than 3 pct of the time. Nine voices pass, and the "derived voice against the voice it copied" table is where to check it: `h_frm_esm`, `h_frm_itm`, `h_frm_itf`, `h_frm_nlf`, `h_frf_nlm`, `h_frf_chf`, `h_frf_dem`, `h_frm_ptf`, `h_frm_ptm`. Each is the same speaker as its donor, so the pair buys prosodic variation, not a new voice.

**Drop the rest.** Two ways to fail the rule, and every remaining derived voice fails one of them. Too fragile: `h_frf_cam` is the worst voice in the pool on every axis (9.3 pct runaway, 31 takes in 332, and a worse median than its donor), followed by `h_frm_nem` at 4.2 pct, `h_frf_nef` and `h_frm_esf` at 3.6 pct, and `h_frf_caf` at 3.3 pct. Pointless: `h_def_frm`, `h_def_hif`, `h_def_him`, `h_frf_arm` and `x_frf_caf_nlm` are stable but read no better than the voices they copied, so the donor is strictly the better choice.

**Clean the text before generating.** Replace U+00A0, U+2009 and U+202F with an ordinary space. The probe measured this as a 4.5x cut in median WER and 16 unusable takes down to 3 on the sentences that carry them, and it is worth more than any voice selection. Leave the guillemets alone, they were measured to be innocent.

**Keep the retry loop on, and expect it to clear about three quarters of the damage.** Runaways are detected from the waveform, not from the ASR, so retrying them with a different seed cannot flatter the benchmark. `driver.sh` runs tts, stt and retry in a loop until no take is suspicious, `stats --attempt first` keeps reporting the unaided number whatever the retries did, and `retries.md` says how well it worked. With retries the fragile hybrids stop being dangerous, but the residue is text-driven and a retry loop cannot fix that.

## What this does not tell you

- **Whether the 37 stuck takes are fixable at all.** Three seeds were not enough for them. Whether a fourth or a tenth would work, or whether those (voice, sentence) pairs are genuinely impossible, is unknown: raising `MAX_ATTEMPTS` is the cheap experiment, but the text fix is the better one if the probe confirms the thin space.
- **How these voices sound.** ECAPA answers "same speaker or not", and WER answers "were the words right". Neither says whether a voice is pleasant, whether a hybrid inherits the donor's prosody or the semantic donor's, or whether the accent survives in a way a listener would notice. That needs ears on a sample.
- **Whether whisper is fair to every voice.** The ASR is one model with its own accent biases, so a foreign-accented voice may be penalised for being correctly transcribable by a human but not by this model. The CER table is the partial check: where WER is high and CER is low, the words were nearly right, which is a scoring artifact more than a generation failure.
