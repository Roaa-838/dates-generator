#!/usr/bin/env bash
# =============================================================================
# scripts/github_setup.sh
# One-shot script to initialise git, create a .gitignore, and push to a
# PRIVATE GitHub repository.
#
# Usage:
#   chmod +x scripts/github_setup.sh
#   ./scripts/github_setup.sh <github_username> <repo_name>
#
# Example:
#   ./scripts/github_setup.sh johndoe date-generation-assignment
#
# Prerequisites:
#   - git installed
#   - GitHub CLI (gh) installed  OR  SSH key added to your GitHub account
#   - Run from the ROOT of the repo (same level as model/ and data/)
# =============================================================================

set -e   # exit on any error

GITHUB_USER="${1:?Usage: $0 <github_username> <repo_name>}"
REPO_NAME="${2:?Usage: $0 <github_username> <repo_name>}"

echo "============================================================"
echo "  GitHub Setup — $GITHUB_USER/$REPO_NAME"
echo "============================================================"

# ── 1. Initialise git (safe to run if already initialised) ──────────────────
if [ ! -d ".git" ]; then
    git init
    echo "[1/7] git init ✓"
else
    echo "[1/7] git already initialised ✓"
fi

# ── 2. Write .gitignore ──────────────────────────────────────────────────────
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/
.conda/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# PyTorch checkpoint files (large — tracked via Git LFS or excluded)
# KEEP best-model weights but ignore raw checkpoint files
model/weights/*_checkpoint.pt
model/weights/*_run_id.txt

# MLflow run data (large; keep only the experiment registry)
mlruns/*/artifacts/
mlruns/*/.trash/

# OS / editor
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp

# Data (may be large; add to LFS if needed)
# data/data.txt   ← uncomment to exclude the raw data file

# Logs & temp
*.log
tmp/
EOF
echo "[2/7] .gitignore written ✓"

# ── 3. Configure git identity (skip if already configured) ──────────────────
if [ -z "$(git config user.email)" ]; then
    git config user.email "${GITHUB_USER}@users.noreply.github.com"
    git config user.name  "${GITHUB_USER}"
fi
echo "[3/7] git identity configured ✓"

# ── 4. Stage everything ──────────────────────────────────────────────────────
git add -A
git status --short
echo "[4/7] files staged ✓"

# ── 5. Initial commit ────────────────────────────────────────────────────────
git diff --cached --quiet || git commit -m "Initial commit: conditional date generation models"
echo "[5/7] initial commit ✓"

# ── 6. Create PRIVATE repo on GitHub and push ────────────────────────────────
if command -v gh &> /dev/null; then
    echo "[6/7] GitHub CLI found — creating private repo..."
    gh repo create "${REPO_NAME}" --private --source=. --remote=origin --push
    echo "[6/7] repo created and pushed via gh ✓"
else
    echo "[6/7] GitHub CLI not found — using SSH remote..."
    REMOTE="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    git remote remove origin 2>/dev/null || true
    git remote add origin "${REMOTE}"
    git branch -M main
    git push -u origin main
    echo "[6/7] pushed to ${REMOTE} ✓"
    echo ""
    echo "  NOTE: Make sure you have created a PRIVATE repo named '${REPO_NAME}'"
    echo "        at https://github.com/new  BEFORE running this step."
fi

# ── 7. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Verification:"
git log --oneline -5
git remote -v
echo ""
echo "============================================================"
echo "  Done!  Your repo is at:"
echo "  https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "============================================================"
