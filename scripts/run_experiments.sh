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
set -euo pipefail

# Modal's own guidance: avoid more than 5 concurrent Volume commits (each
# train() run ends with results_volume.commit()) to avoid commit contention.
MAX_CONCURRENT=5
CMDS=$(mktemp)
trap 'rm -f "$CMDS"' EXIT

# 1. Prediction target sweep (dino_wm/wall, nanowm_s2)
G=pred_target_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for pred in v x; do
  echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=$pred;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[$pred]\"" >> "$CMDS"
done
# No extra overrides for flow: src/diffusion/flow_matching.py's FlowMatching
# class ignores noise_schedule/zero_terminal_snr entirely (never receives the
# computed betas) and snr_gamma is explicitly a no-op ("unused, kept for API
# compat") -- it always uses plain unweighted MSE against u = x_0 - eps.
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=flow;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[flow]\"" >> "$CMDS"
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=linear;experiment.diffusion.zero_terminal_snr=false;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[epsilon]\"" >> "$CMDS"

# 2. Action injection (dino_wm/pusht, nanowm_s2)
G=injection_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for inj in additive film adaln adaln_fuse cross_attention; do
  echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/pusht --max-steps 2000 --group $G --overrides \"model.action_injection.type=$inj;wandb.tags=[$inj]\"" >> "$CMDS"
done

# 3. Model scale (dino_wm/wall, nanowm_s2 vs nanowm_b2)
G=scale_sweep_$(date +%m-%d-%Y_%H-%M-%S)
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G" >> "$CMDS"
echo "modal run modal_train.py --model nanowm_b2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G" >> "$CMDS"

# 4. Noise schedule / ZTSNR (dino_wm/wall, nanowm_s2)
G=schedule_sweep_$(date +%m-%d-%Y_%H-%M-%S)
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=squaredcos_cap_v2;experiment.diffusion.zero_terminal_snr=true;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[cosine_ztsnr_eps]\"" >> "$CMDS"
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.diffusion.pred_name=epsilon;experiment.diffusion.noise_schedule=linear;experiment.diffusion.zero_terminal_snr=false;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[linear_eps]\"" >> "$CMDS"

# 5. Gradient clipping (dino_wm/wall, nanowm_s2)
G=clip_sweep_$(date +%m-%d-%Y_%H-%M-%S)
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.training.gradient_clip_start_step=0;experiment.training.gradient_clip_norm=0.1;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[clip_on]\"" >> "$CMDS"
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 4000 --group $G --overrides \"experiment.training.gradient_clip_start_step=0;experiment.training.gradient_clip_norm=1e9;experiment.training.optimizer.lr_warmup_steps=200;wandb.tags=[clip_off]\"" >> "$CMDS"

# 6. Timestep sampling (dino_wm/wall, nanowm_s2)
G=timestep_sweep_$(date +%m-%d-%Y_%H-%M-%S)
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G --overrides \"experiment.diffusion.timestep_sampling=uniform;wandb.tags=[uniform]\"" >> "$CMDS"
echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/wall --max-steps 2000 --group $G --overrides \"experiment.diffusion.timestep_sampling=logit_normal;wandb.tags=[logit_normal]\"" >> "$CMDS"

# 7. Dataset comparison (nanowm_s2, across point_maze / wall / pusht)
G=dataset_sweep_$(date +%m-%d-%Y_%H-%M-%S)
for ds in wall point_maze pusht; do
  echo "modal run modal_train.py --model nanowm_s2 --experiment quickstart --dataset dino_wm/$ds --max-steps 2000 --group $G --overrides \"wandb.tags=[$ds]\"" >> "$CMDS"
done

n_cmds=$(wc -l < "$CMDS" | tr -d ' ')
echo "Queued $n_cmds runs across 7 experiments, $MAX_CONCURRENT concurrent." >&2

if [[ "${1:-}" == "--dry-run" ]]; then
  cat "$CMDS"
  exit 0
fi

# Not `xargs -I`: macOS/BSD xargs caps the -I replacement at ~255 bytes and
# fails outright ("command line cannot be assembled, too long") on our
# --overrides lines, which run 270-330 chars. A plain throttled loop has no
# such limit and works identically on bash/zsh, Linux/macOS.
while IFS= read -r cmd; do
  # </dev/null: without this, each backgrounded job inherits the loop's own
  # stdin (fd 0, redirected from $CMDS below) via fork(), sharing its read
  # offset. If any job ever reads stdin, it silently consumes/corrupts the
  # remaining command list the loop is still reading.
  eval "$cmd" </dev/null &
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
wait
