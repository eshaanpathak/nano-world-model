"""Train Nano World Model (NanoWM-S/2, DINO-WM PushT) on Modal.

Usage (see README section below for one-time setup commands):
    modal run modal_train.py
    modal run modal_train.py --model nanowm_b2 --max-steps 100000
"""

import modal

app = modal.App("nano-world-model")

REPO_URL = "https://github.com/eshaanpathak/nano-world-model.git"
# Personal deploy branch: mirrors upstream main plus whatever feature
# branches have been merged in for my own Modal runs. Rebase this branch
# onto main (kept in sync with upstream) to pick up new merged work.
REPO_BRANCH = "personal"

# NOTE: this clones a fixed branch from GitHub at image-build time, so
# uncommitted local changes are NOT included. Push your changes first, or
# swap this for `modal.Image.add_local_dir(".", "/repo", copy=True)` to ship
# the local tree instead (that will also pull in any large local
# data/results directories, so add an `ignore=` filter if you do).
# CI pins uv 0.12.0. A newer uv will often try to rewrite uv.lock (new
# lock revision / marker fields) and then `uv sync --locked` fails even
# though the committed lock is valid. `--frozen` installs that lock as-is.
# Use an absolute uv path; Modal does not expand ${PATH} in .env() values.
UV_VERSION = "0.12.0"
UV_BIN = "/root/.local/bin/uv"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .run_commands(
        f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh"
    )
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        }
    )
    .run_commands(f"git clone --depth 1 --branch {REPO_BRANCH} {REPO_URL} /repo")
    .workdir("/repo")
    .run_commands(f"{UV_BIN} --version && {UV_BIN} sync --frozen --managed-python")
    # The pinned wandb==0.21.1 crashes in a legacy, non-essential codepath
    # (server.py's `viewer` query, used only to print "logged in as <user>")
    # when the current wandb.ai backend returns `flags: null` — a client-side
    # SDK bug unrelated to the rest of the locked stack. Upgrade just wandb.
    .run_commands(f"{UV_BIN} pip install -U wandb")
)

data_volume = modal.Volume.from_name("nanowm-data", create_if_missing=True)
results_volume = modal.Volume.from_name("nanowm-results", create_if_missing=True)

DATASET_DIR = "/data/dino_wm_data"
RESULTS_DIR = "/results"

# Tiny, GPU-free image/function just for downloading and unzipping datasets
# onto the volume. `modal shell data_shell` starts in a few seconds instead
# of the ~2min it takes to schedule a GPU container for `train`.
data_prep_image = modal.Image.debian_slim(python_version="3.11").apt_install("curl", "unzip")


@app.function(image=data_prep_image, volumes={"/data": data_volume}, timeout=60 * 60)
def data_shell():
    pass


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    model: str = "nanowm_s2",
    experiment: str = "dino_wm_pusht",
    dataset: str = "dino_wm/pusht",
    max_steps: int = 100_000,
    batch_size: int = 8,
    group: str = None,
):
    import os
    import subprocess

    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not wandb_api_key:
        raise RuntimeError(
            "WANDB_API_KEY is empty or missing from the container environment. "
            "Check `modal secret list` for a secret named 'wandb-secret' with a "
            "key literally named WANDB_API_KEY, and that this function's "
            "secrets=[...] references it."
        )

    env = {
        **os.environ,
        "WANDB_API_KEY": wandb_api_key,
        "DATASET_DIR": DATASET_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "WANDB_PROJECT": "nano-world-model",
        "WANDB_ENTITY": "eshaanpathak-ai-team",
    }

    args = [
        "/repo/.venv/bin/python",
        "src/main.py",
        f"experiment={experiment}",
        f"dataset={dataset}",
        f"model={model}",
        f"experiment.training.max_steps={max_steps}",
        f"experiment.training.batch_size={batch_size}",
        "wandb.enabled=true",
    ]
    if group:
        args.append(f"wandb.group={group}")

    subprocess.run(args, cwd="/repo", env=env, check=True)

    results_volume.commit()


@app.local_entrypoint()
def main(
    model: str = "nanowm_s2",
    experiment: str = "dino_wm_pusht",
    dataset: str = "dino_wm/pusht",
    max_steps: int = 100_000,
    batch_size: int = 8,
    group: str = None,
):
    train.remote(
        model=model,
        experiment=experiment,
        dataset=dataset,
        max_steps=max_steps,
        batch_size=batch_size,
        group=group,
    )
