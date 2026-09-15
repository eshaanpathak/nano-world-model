#!/usr/bin/env bash
# Modal commands for the 7 toy-scale experiments discussed in chat.
# GPU: L4 (hardcoded default in modal_train.py). Model: nanowm_s2 except
# experiment 3 (model scale), which compares nanowm_s2 vs nanowm_b2.
#
# Experiments 1, 4, 5 (pred target, noise schedule/ZTSNR, gradient clipping)
# run at 4000 steps with lr_warmup_steps cut from the default 1000 to 200:
# these are the stability/divergence-focused experiments, and default
# warmup would otherwise burn half the toy step budget suppressing the very
# gradient dynamics being tested. Experiments 2, 3, 6, 7 stay at 2000 steps
# / default warmup since a longer post-warmup window doesn't help them.
#
# Usage:
#   ./run_experiments.sh            # queue all 20 runs, 5 concurrent
#   ./run_experiments.sh --dry-run  # just print the commands, don't run them
#
# Each run's stdout/stderr goes to its own file under logs/<run-timestamp>/
# (concurrent `modal run` output interleaved on one terminal is unreadable),
# alongside commands.tsv (the exact command queued for each label). After all
# runs finish (or you Ctrl-C), a summary prints to stderr and is written to
# logs/<run-timestamp>/summary.log, breaking runs into: succeeded, FAILED,
# INTERRUPTED (killed mid-run by Ctrl-C), and NOT STARTED (still queued when
# you hit Ctrl-C). A nonzero script exit code means something didn't
# succeed cleanly (a failure, or you interrupting it).
#
# Ctrl-C stops both the queueing loop AND every already-launched run: the
# INT trap below SIGTERMs the tracked child PIDs directly. Plain Ctrl-C
# would NOT do this on its own -- bash sets SIGINT to be ignored for any
# `cmd &` started from a non-interactive script, precisely so background
# jobs survive a Ctrl-C aimed at the foreground script. SIGTERM isn't
# subject to that ignore-by-default rule, so it reaches the children (their
# own `modal run` process, and whatever it forked) even though SIGINT
# wouldn't. This is safe to do at any point: each run writes to its own
# timestamped hydra dir (never shared across runs) and only touches the
# shared results volume via commit() after finishing locally, so a run
# killed mid-flight just never gets exposed there -- it can't corrupt or
# collide with any other run's output.
set -euo pipefail

# A long unattended sweep can outlast the terminal's idle-sleep window --
# once the Mac sleeps, the network drops out from under every in-flight
# `modal run`, and the local process for each run loses its connection even
# though `--detach` (below) keeps the remote job itself alive on Modal. That
# desyncs this script's own bookkeeping: it decides success/failure and
# concurrency slots from the *local* `modal run` exit status, so a
# disconnect makes it report a still-running remote job as FAILED and queue
# a duplicate run in the freed "slot". `caffeinate -i` keeps the Mac (and
# therefore the network) awake for the script's lifetime so that never
# happens; re-exec under a guard var so this only wraps once.
if [[ -z "${RUN_EXPERIMENTS_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export RUN_EXPERIMENTS_CAFFEINATED=1
  exec caffeinate -i "$0" "$@"
fi

# Modal's own guidance: avoid more than 5 concurrent Volume commits (each
# train() run ends with results_volume.commit()) to avoid commit contention.
MAX_CONCURRENT=5
CMDS=$(mktemp)
trap 'rm -f "$CMDS"' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="$SCRIPT_DIR/logs/$RUN_TS"

# Recursively TERMs a pid and every descendant, deepest first. Not `kill
# -TERM "${PIDS[@]}"` in on_interrupt below: that only reaches the tracked
# `eval "$cmd" &` subshells, not whatever they fork underneath (verified --
# a compound eval'd command left its grandchild running, orphaned, after
# its immediate parent died). Not `kill -TERM -- -$$` (whole process group)
# either: that assumes this script's pid is also its process group's
# leader, which isn't guaranteed by how it might be invoked (verified false
# under at least one wrapper) -- get it wrong and you either miss processes
# or, worse, signal something outside this script's own job entirely. A
# plain parent->child walk has no such assumption.
kill_tree() {
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

# Buckets shared by the normal end-of-run summary and the Ctrl-C summary
# below, so both print in the same format via print_summary().
PIDS=()
LABELS=()
OK_LABELS=()
FAILED_LABELS=()
INTERRUPTED_LABELS=()
NOT_STARTED_LABELS=()

print_summary() {
  local status_suffix=$1  # "" for a normal finish, " -- INTERRUPTED" for Ctrl-C
  {
    echo "== run_experiments.sh summary ($RUN_TS)$status_suffix =="
    echo "${#OK_LABELS[@]}/$n_cmds runs succeeded."
    if [ "${#FAILED_LABELS[@]}" -gt 0 ]; then
      echo "FAILED (${#FAILED_LABELS[@]}):"
      for label in "${FAILED_LABELS[@]}"; do
        echo "  - $label  (log: $LOG_DIR/*_${label}.log)"
      done
    fi
    if [ "${#INTERRUPTED_LABELS[@]}" -gt 0 ]; then
      echo "INTERRUPTED (${#INTERRUPTED_LABELS[@]}): ${INTERRUPTED_LABELS[*]}"
    fi
    if [ "${#NOT_STARTED_LABELS[@]}" -gt 0 ]; then
      echo "NOT STARTED (${#NOT_STARTED_LABELS[@]}): ${NOT_STARTED_LABELS[*]}"
    fi
  } | tee "$LOG_DIR/summary.log" >&2
}

on_interrupt() {
  echo "" >&2
  if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "Ctrl-C: no runs had started yet." >&2
    exit 130
  fi

  echo "Ctrl-C: stopping ${#PIDS[@]} in-flight run(s)..." >&2
  for pid in "${PIDS[@]}"; do
    kill_tree "$pid"
  done

  # A job that had already finished (fast run, race with Ctrl-C) still
  # reports its real exit status here -- kill_tree on an already-dead pid is
  # a harmless no-op. 143 = 128+SIGTERM is what `wait` reports for a job we
  # just killed; anything else nonzero is a genuine failure that happened
  # before the interrupt, not something Ctrl-C caused.
  for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    label=${LABELS[$i]}
    if wait "$pid"; then
      OK_LABELS+=("$label")
    else
      status=$?
      if [ "$status" -eq 143 ]; then
        INTERRUPTED_LABELS+=("$label")
      else
        FAILED_LABELS+=("$label")
      fi
    fi
  done

  if [ "$idx" -lt "$n_cmds" ]; then
    while IFS=$'\t' read -r label _; do
      NOT_STARTED_LABELS+=("$label")
    done < <(tail -n "+$((idx + 1))" "$CMDS")
  fi

  print_summary " -- INTERRUPTED"
  exit 130
}
trap on_interrupt INT

# Each queued line is "label<TAB>command" -- the label (sweep name + arm)
# is used for the per-run log filename and the failure summary, so a failed
# run can be identified without re-parsing its --overrides string.
enqueue() {
  printf '%s\t%s\n' "$1" "$2" >> "$CMDS"
}

# 1. Prediction target sweep (dino_wm/wall, nanowm_s2)
G=pred_target_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for pred in v x; do
  enqueue "pred_target_sweep-$pred" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=$pred;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[$pred]\""
done
# No extra overrides for flow: src/diffusion/flow_matching.py's FlowMatching
# class ignores noise_schedule/zero_terminal_snr entirely (never receives the
# computed betas) and snr_gamma is explicitly a no-op ("unused, kept for API
# compat") -- it always uses plain unweighted MSE against u = x_0 - eps.
enqueue "pred_target_sweep-flow" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=flow;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[flow]\""
enqueue "pred_target_sweep-epsilon" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=linear;experiment.diffusion.zero_terminal_snr=false;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[epsilon]\""

# 2. Action injection (dino_wm/pusht, nanowm_s2)
G=injection_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for inj in additive film adaln adaln_fuse cross_attention; do
  enqueue "injection_sweep-$inj" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/pusht --max-steps 2000 --group $G --overrides \"model.action_injection.type=$inj;wandb.tags=[$inj]\""
done

# 3. Model scale (dino_wm/wall, nanowm_s2 vs nanowm_b2)
G=scale_sweep_$(date +%m-%d-%Y_%H-%M-%S)
enqueue "scale_sweep-nanowm_s2" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G"
enqueue "scale_sweep-nanowm_b2" "modal run --detach modal_train.py --model nanowm_b2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G"

# 4. Noise schedule / ZTSNR (dino_wm/wall, nanowm_s2)
G=schedule_sweep_$(date +%m-%d-%Y_%H-%M-%S)
enqueue "schedule_sweep-cosine_ztsnr_eps" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=squaredcos_cap_v2;experiment.diffusion.zero_terminal_snr=true;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[cosine_ztsnr_eps]\""
enqueue "schedule_sweep-linear_eps" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=linear;experiment.diffusion.zero_terminal_snr=false;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[linear_eps]\""

# 5. Gradient clipping (dino_wm/wall, nanowm_s2)
G=clip_sweep_$(date +%m-%d-%Y_%H-%M-%S)
enqueue "clip_sweep-clip_on" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.training.gradient_clip_start_step=0;experiment.training.gradient_clip_norm=0.1;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[clip_on]\""
enqueue "clip_sweep-clip_off" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.training.gradient_clip_start_step=0;experiment.training.gradient_clip_norm=1e9;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[clip_off]\""

# 6. Timestep sampling (dino_wm/wall, nanowm_s2)
G=timestep_sweep_$(date +%m-%d-%Y_%H-%M-%S)
enqueue "timestep_sweep-uniform" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G --overrides \"experiment.diffusion.timestep_sampling=uniform;wandb.tags=[uniform]\""
enqueue "timestep_sweep-logit_normal" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G --overrides \"experiment.diffusion.timestep_sampling=logit_normal;wandb.tags=[logit_normal]\""

# 7. Dataset comparison (nanowm_s2, across point_maze / wall / pusht)
G=dataset_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for ds in wall point_maze pusht; do
  enqueue "dataset_sweep-$ds" "modal run --detach modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/$ds --max-steps 2000 --group $G --overrides \"wandb.tags=[$ds]\""
done

n_cmds=$(wc -l < "$CMDS" | tr -d ' ')
echo "Queued $n_cmds runs across 7 experiments, $MAX_CONCURRENT concurrent." >&2

if [[ "${1:-}" == "--dry-run" ]]; then
  cut -f2- "$CMDS"
  exit 0
fi

mkdir -p "$LOG_DIR"
cp "$CMDS" "$LOG_DIR/commands.tsv"  # $CMDS itself is deleted by the EXIT trap
echo "Per-run logs: $LOG_DIR" >&2

# Not `xargs -I`: macOS/BSD xargs caps the -I replacement at ~255 bytes and
# fails outright ("command line cannot be assembled, too long") on our
# --overrides lines, which run 270-330 chars. A plain throttled loop has no
# such limit and works identically on bash/zsh, Linux/macOS.
idx=0
while IFS=$'\t' read -r label cmd; do
  idx=$((idx + 1))
  job_log="$LOG_DIR/$(printf '%02d' "$idx")_${label}.log"
  # </dev/null: without this, each backgrounded job inherits the loop's own
  # stdin (fd 0, redirected from $CMDS below) via fork(), sharing its read
  # offset. If any job ever reads stdin, it silently consumes/corrupts the
  # remaining command list the loop is still reading.
  eval "$cmd" </dev/null >"$job_log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$label")
  echo "[$idx/$n_cmds] started $label (pid $!) -> $job_log" >&2
  # Several arms share identical (model, num_frames, dataset) settings, and
  # hydra's run-dir name (src/configs/config.yaml) is only unique to the
  # second for a given combo -- stagger launches so two same-combo runs
  # landing in the same wall-clock second, and colliding on the shared
  # Modal results volume, is very unlikely.
  sleep 10
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_CONCURRENT" ]; do
    sleep 1
  done
done < "$CMDS"

# Collect exit codes individually (plain `wait` only reports the status of
# the last job waited on) so we can report exactly which runs failed.
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    OK_LABELS+=("${LABELS[$i]}")
  else
    FAILED_LABELS+=("${LABELS[$i]}")
  fi
done

print_summary ""

if [ "${#FAILED_LABELS[@]}" -gt 0 ]; then
  exit 1
fi
