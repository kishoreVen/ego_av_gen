#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="3.12"

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: 'uv' is not installed. Please install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

create_venv() {
    if [ -d "$VENV_DIR" ]; then
        read -r -p "A venv already exists at ${VENV_DIR}. Delete and recreate? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo "Removing existing venv..."
                rm -rf "$VENV_DIR"
                ;;
            *)
                echo "Keeping existing venv."
                return
                ;;
        esac
    fi
    echo "Creating Python ${PYTHON_VERSION} venv with uv at ${VENV_DIR}..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --prompt scene_gen
    echo "Venv created."
}

install_requirements() {
    echo "Installing requirements..."
    uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"
    echo "Requirements installed."
}

install_git_deps() {
    local THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
    mkdir -p "$THIRD_PARTY_DIR"

    # SAM3 (requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+)
    if [ ! -d "$THIRD_PARTY_DIR/sam3" ]; then
        echo "Cloning SAM3..."
        git clone https://github.com/facebookresearch/sam3.git "$THIRD_PARTY_DIR/sam3"
    fi
    echo "Installing SAM3..."
    uv pip install --python "$VENV_DIR/bin/python" -e "$THIRD_PARTY_DIR/sam3"
}

main() {
    check_uv
    create_venv
    install_requirements
    install_git_deps
    echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
}

main "$@"
