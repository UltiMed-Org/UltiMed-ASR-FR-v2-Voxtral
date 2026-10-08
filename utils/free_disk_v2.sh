#!/usr/bin/env bash
# Deletes only what was explicitly approved on 2026-10-08. Written with Claude Code.
#
# Dry run by default: prints every path and its size and removes nothing.
# Pass --go to actually delete. Nothing here needs sudo (all paths are yours).
#
# NOT included, deliberately: hf_parquet (you want the spare copy), united_syn_med
# (an external English dataset, not a duplicate), and fleurs (6 of its 25 languages
# are live validation manifests in training_config.yaml).
set -euo pipefail
GO=0; [[ "${1:-}" == "--go" ]] && GO=1
N=/media/$USER/4TO_DRIVE/nemo_experiments/parakeet-tdt-ultimed-french-medical
G=/media/$USER/4TO_DRIVE/voxtral_grid
D=/media/$USER/ssd1TO/audio_datasets
total=0
drop() {  # drop <path>...
  for p in "$@"; do
    [[ -e "$p" ]] || continue
    local kb; kb=$(du -sk "$p" | cut -f1); total=$((total + kb))
    if (( GO )); then rm -rf -- "$p"; printf 'deleted  %8s  %s\n' "$(numfmt --to=iec $((kb*1024)))" "$p"
    else printf 'would delete  %8s  %s\n' "$(numfmt --to=iec $((kb*1024)))" "$p"; fi
  done
}

# 1. Runs 1.1.0 to 1.5.0: keep each run's exported .nemo, drop the 6 intermediate .ckpt.
for v in 1.1.0 1.2.0 1.3.0 1.3.0-nospd 1.4.0 1.5.0; do
  [[ -d "$N/$v" ]] || continue
  while IFS= read -r -d '' f; do drop "$f"; done < <(find "$N/$v" -name '*.ckpt' -print0)
done

# 2. Run 1.6.0: keep the .nemo and the single best checkpoint, drop the rest. "Best" is the
# lowest combined_macro_val_wer in the filename; -last is a duplicate of the final step.
if [[ -d "$N/1.6.0" ]]; then
  # Lowest val WER wins; on a tie the HIGHER step wins, because 1.6.0 has two checkpoints
  # at 0.0687 (steps 7000 and 8000) and keeping the less-trained one would be a silent loss.
  best=$(find "$N/1.6.0" -name '*.ckpt' ! -name '*-last.ckpt' -printf '%f\n' \
         | sed -E 's/.*step=([0-9]+).*val_wer=([0-9.]+)\.ckpt/\2 \1 &/' \
         | sort -k1,1n -k2,2nr | head -1 | cut -d' ' -f3)
  echo "# 1.6.0: keeping best checkpoint $best"
  while IFS= read -r -d '' f; do [[ "$(basename "$f")" == "$best" ]] || drop "$f"; done \
    < <(find "$N/1.6.0" -name '*.ckpt' -print0)
fi

# 3. Run 1.7.0: the averaged/ and ckpt_archive/ trees are derived from checkpoints/ and
# rebuildable, so they go; checkpoints/ and export/ stay untouched.
drop "$N/1.7.0/averaged" "$N/1.7.0/ckpt_archive"

# 4. The French grid's audio. The selection is finished; ledger.jsonl, embeddings.npz and
# the reports stay, and they are what every conclusion was drawn from.
drop "$G/GRID_fr_fr/wav"

# 5. de_de (779 takes, never scored) and the empty dirs of the abandoned languages.
drop "$G/GRID_de_de"
for l in es_419 it_it nl_nl pt_br ar_eg hi_in; do drop "$G/GRID_$l"; done

# 6. Superseded manifest backups.
drop "$D"/NeMO_files.bak-*

printf '\n%s: %s\n' "$( ((GO)) && echo freed || echo would free )" "$(numfmt --to=iec $((total*1024)))"
(( GO )) || echo "re-run with --go to delete"
