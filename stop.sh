#!/usr/bin/env bash
# Emergency hold; --disable additionally disables arms and independent grippers.
set -eo pipefail
PLATFORM_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$PLATFORM_DIR/env.sh"
python - "$@" <<'PY'
import argparse,json
from nero_eval_workbench.common import new_session,write
from nero_eval_workbench.worker import stop_services
parser=argparse.ArgumentParser();parser.add_argument('--disable',action='store_true');args=parser.parse_args()
directory=new_session('emergency_stop');result=stop_services(args.disable)
write(directory/'stop_result.json',result)
print(directory);print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(2 if result['errors'] else 0)
PY
