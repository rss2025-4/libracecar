set -e
set -u
set -x

export PYTHONUNBUFFERED=1

python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

cd /tmp_poetry
~/.local/bin/poetry install
