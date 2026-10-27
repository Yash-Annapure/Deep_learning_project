#!/usr/bin/env bash
# Run this on the server after the container restarts (a restart wipes the home folder):
#     bash server_setup.sh
# It is safe to run any time; it only repairs what is missing.
cd "$(dirname "$0")"

echo "== 1. Python environment"
uv sync

echo "== 2. Earth Engine login"
if [ -f ~/vault/.earthengine/credentials ]; then
  mkdir -p ~/.config/earthengine && cp ~/vault/.earthengine/credentials ~/.config/earthengine/
  echo "restored from ~/vault/.earthengine/"
else
  echo "no saved login yet. Do this once, then run this script again:"
  echo "  uv run earthengine authenticate --auth_mode=notebook --scopes https://www.googleapis.com/auth/earthengine"
  echo "  mkdir -p ~/vault/.earthengine && cp ~/.config/earthengine/credentials ~/vault/.earthengine/"
fi

echo "== 2b. NASA Earthdata login (coherence via ASF HyP3)"
if [ -f ~/vault/.netrc ]; then
  cp ~/vault/.netrc ~/.netrc && chmod 600 ~/.netrc
  echo "restored from ~/vault/.netrc"
else
  echo "no saved Earthdata login yet (see the coherence notebook for the one-time step)"
fi

echo "== 3. Jupyter kernel"
.venv/bin/python -m ipykernel install --user --name deep-learning-project --display-name "Python (deep-learning-project)"

echo "== 4. Checks"
.venv/bin/python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT available')"
.venv/bin/python -c "import ee; ee.Initialize(project='deep-learning-project-512013'); print('Earth Engine: logged in')" 2>/dev/null \
  || echo "Earth Engine: NOT logged in (see step 2)"
.venv/bin/python -c "import hyp3_sdk; hyp3_sdk.HyP3(); print('Earthdata / HyP3: logged in')" 2>/dev/null \
  || echo "Earthdata / HyP3: NOT logged in (see step 2b)"
[ -x ~/vault/bin/cloudflared ] && echo "cloudflared: found in ~/vault/bin" || echo "cloudflared: missing, download it into ~/vault/bin"
