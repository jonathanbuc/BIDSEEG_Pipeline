#!/usr/bin/env bash
# Resumable per-subject EEG run: for each subject, set participants.tsv to just that
# subject, run a->b->c, and append its per-trial alpha to EEG_iaf_master.csv. Subjects
# already present in the master are skipped, so a killed run just resumes with `--rest`.
#
# Usage:
#   bash run_eeg_incremental.sh 003 004 005      # specific subjects
#   bash run_eeg_incremental.sh --rest           # every full-42 subject not yet in master
set -uo pipefail

CFG=inputs_openneuro.json
D="openneuro data/default/derivatives/BIDSprocessed"
FULL="$D/participants_full42.tsv"
RAW="openneuro data/default/participants.tsv"
PROC="$D/participants.tsv"
SP="$D/results/SpectralParameterization"
MASTER="$SP/EEG_iaf_master.csv"
LOG="$SP/incremental_progress.log"

# resolve subject list
if [ "${1:-}" = "--rest" ]; then
  SUBJECTS=$(tail -n +2 "$FULL" | sed -E 's/^sub-([0-9]+).*/\1/')
else
  SUBJECTS="$*"
fi

HDR=$(head -1 "$FULL")
for s in $SUBJECTS; do
  if [ -f "$MASTER" ] && grep -q "sub-$s" "$MASTER"; then
    echo "$(date +%H:%M:%S) skip sub-$s (already in master)" | tee -a "$LOG"; continue
  fi
  row=$(grep -P "^sub-$s\t" "$FULL")
  if [ -z "$row" ]; then echo "no row for sub-$s" | tee -a "$LOG"; continue; fi
  printf '%s\n%s\n' "$HDR" "$row" > "$RAW"
  printf '%s\n%s\n' "$HDR" "$row" > "$PROC"

  echo "$(date +%H:%M:%S) START sub-$s" | tee -a "$LOG"
  conda run -n mne-env python a_preprocessing_module.py "$CFG"  > "$SP/_last_a.log" 2>&1 || { echo "  a FAIL sub-$s" | tee -a "$LOG"; continue; }
  conda run -n mne-env python b_ArtifactCorrection_module.py "$CFG" > "$SP/_last_b.log" 2>&1 || { echo "  b FAIL sub-$s" | tee -a "$LOG"; continue; }
  conda run -n mne-env python c_EEGAnalysis_module.py "$CFG" > "$SP/_last_c.log" 2>&1 || { echo "  c FAIL sub-$s" | tee -a "$LOG"; continue; }

  if [ -f "$MASTER" ]; then tail -n +2 "$SP/EEG_iaf.csv" >> "$MASTER"; else cp "$SP/EEG_iaf.csv" "$MASTER"; fi
  n=$(tail -n +2 "$MASTER" | cut -d, -f2 | sort -u | wc -l)   # col 2 = participant
  echo "$(date +%H:%M:%S) DONE sub-$s  (master now $n subjects)" | tee -a "$LOG"
done
echo "$(date +%H:%M:%S) RUN COMPLETE" | tee -a "$LOG"
