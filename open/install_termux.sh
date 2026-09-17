#!/data/data/com.termux/files/usr/bin/bash
# install_termux.sh - one-shot installer for opencode_py on Termux (armv7 / arm64)
#
# Run on the phone:
#   pkg install -y git
#   git clone <YOUR_REPO_URL> opencode_py
#   cd opencode_py && bash install_termux.sh
#   opencode-py            # TUI; then /connect and paste your free Zen key
set -e

# ---------------------------------------------------------------------------
# Where is your fork of opencode_py? Override with:
#   REPO_URL=https://github.com/<you>/opencode_py.git bash install_termux.sh
# Defaults to the upstream repo when not set.
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/anomalyco/opencode_py.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/opencode_py}"

echo "==> Updating package lists"
pkg update -y

echo "==> Installing system packages"
pkg install -y python git ripgrep openssh android-tools

if [ -d "$INSTALL_DIR/.git" ]; then
  echo "==> Updating existing checkout in $INSTALL_DIR"
  cd "$INSTALL_DIR"
  git pull --ff-only || true
else
  echo "==> Cloning opencode_py"
  git clone "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi

echo "==> Installing Python dependencies (pure-Python, armv7-safe)"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install .

echo
echo "===================================================================="
echo " Installation complete."
echo "===================================================================="
echo
echo " Next steps:"
echo "   1. Get a FREE Zen API key at https://opencode.ai/auth"
echo "   2. Run:   opencode-py"
echo "   3. Type  /connect  -> OpenCode Zen -> paste your key"
echo "   4. Optional providers: /connect Groq / Cerebras / Google /"
echo "      OpenRouter / NVIDIA / Mistral / GitHub (each needs its own key)"
echo "   5. Headless fallback for low-RAM:"
echo "        echo 'hi' | opencode-py --no-tui"
echo
echo " Phone-Chrome control (browser tool, optional one-time setup):"
echo "   1. Android Settings > Developer options > Wireless debugging ON"
echo "   2. Pair with pairing code, then in Termux:"
echo "        adb pair <host>:<port> <code>"
echo "        adb forward tcp:9222 localabstract:chrome_devtools_remote"
echo "   3. Open Chrome once, then inside opencode-py run:"
echo "        browser action=connect   (or status to check)"
echo
echo " Low-RAM tips:"
echo "   - Use *-free / flash models (the default free list)"
echo "   - Prefix a long session with /clear or /new to trim context"
echo "   - If output looks stuck, run:  PYTHONUNBUFFERED=1 opencode-py"
echo
echo " Troubleshooting: see the README's troubleshooting section."
