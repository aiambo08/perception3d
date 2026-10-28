# Sourced by WSL entrypoints. Venv lives on the Linux filesystem ($HOME),
# never under /mnt/c (OneDrive) — torch/TRT installs there stall or fail.
#
#   source scripts/wsl_env.sh
#   bash scripts/run_bdd_eval_wsl.sh
#
# Canonical workflow: open an Ubuntu WSL terminal (not PowerShell) and work there.
export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/perception3d}"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"

# Dataset default (Windows disk is OK for data; code/venv should be Linux).
export BDD="${BDD:-/mnt/c/Users/niaib/datasets/bdd100k}"
