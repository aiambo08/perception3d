#!/usr/bin/env bash
# F7 dual-rate loop on KITTI or a video, with the DoD verdicts in a JSON report.
#
#   bash scripts/run_pipeline.sh kitti-tracking <root> <seq> [extra run_pipeline.py args...]
#   bash scripts/run_pipeline.sh kitti-raw      <drive_dir>  [extra args...]
#   bash scripts/run_pipeline.sh video          <file.mp4>   [extra args...]
#   bash scripts/run_pipeline.sh cpu            <root> <seq> [extra args...]   # GT boxes, no GPU
#
# Environment (all optional):
#   DET_ENGINE    detector engine   (default models/detector_1024x320_fp16.engine)
#   DEPTH_ENGINE  depth engine      (default models/depth_924x280_fp16.engine; NONE = no depth)
#   HZ            input rate        (default 60)
#   DURATION      seconds           (default 300 = the DoD's 5 min; implies --loop)
#   TELEMETRY     none|null|jsonl|rerun (default rerun with --rerun-spawn; set RERUN_CONNECT
#                 to attach to a running viewer, RERUN_SAVE=file.rrd to record instead)
#   REPORT        JSON path         (default reports/f7_<mode>.json)
#   CAPTURE_THREAD=1 uses the capture-thread variant (LatestFrameSlot) instead of the paced loop.
#
# Examples:
#   DURATION=60 bash scripts/run_pipeline.sh kitti-tracking "$KT" 0001
#   TELEMETRY=none bash scripts/run_pipeline.sh kitti-tracking "$KT" 0001 --ego oxts
#   DEPTH_ENGINE=NONE TELEMETRY=jsonl bash scripts/run_pipeline.sh cpu "$KT" 0001 --duration 20
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PY=${PY:-"uv run --no-sync python"}

MODE=${1:?mode: kitti-tracking | kitti-raw | video | cpu}
shift

DET_ENGINE=${DET_ENGINE:-$ROOT/models/detector_1024x320_fp16.engine}
DEPTH_ENGINE=${DEPTH_ENGINE:-$ROOT/models/depth_924x280_fp16.engine}
HZ=${HZ:-60}
DURATION=${DURATION:-300}
TELEMETRY=${TELEMETRY:-rerun}
CAPTURE_THREAD=${CAPTURE_THREAD:-0}

ARGS=(--hz "$HZ" --duration "$DURATION" --loop --telemetry "$TELEMETRY")
[[ "$CAPTURE_THREAD" == "1" ]] && ARGS+=(--capture-thread)
if [[ "$TELEMETRY" == "rerun" ]]; then
  if [[ -n "${RERUN_SAVE:-}" ]]; then ARGS+=(--rerun-save "$RERUN_SAVE")
  elif [[ -n "${RERUN_CONNECT:-}" ]]; then ARGS+=(--rerun-connect "$RERUN_CONNECT")
  else ARGS+=(--rerun-spawn); fi
fi

case "$MODE" in
  kitti-tracking)
    KT_ROOT=${1:?kitti tracking root}; SEQ=${2:?sequence id}; shift 2
    ARGS+=(--kitti-tracking "$KT_ROOT" --seq "$SEQ" --boxes detector --det-engine "$DET_ENGINE")
    ;;
  kitti-raw)
    DRIVE=${1:?kitti raw drive dir}; shift
    ARGS+=(--kitti-raw "$DRIVE" --boxes detector --det-engine "$DET_ENGINE")
    ;;
  video)
    VIDEO=${1:?video file}; shift
    ARGS+=(--video "$VIDEO" --boxes detector --det-engine "$DET_ENGINE")
    ;;
  cpu)
    KT_ROOT=${1:?kitti tracking root}; SEQ=${2:?sequence id}; shift 2
    ARGS+=(--kitti-tracking "$KT_ROOT" --seq "$SEQ" --boxes gt)
    DEPTH_ENGINE=NONE
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

[[ "$DEPTH_ENGINE" != "NONE" ]] && ARGS+=(--depth-engine "$DEPTH_ENGINE")

REPORT=${REPORT:-$ROOT/reports/f7_${MODE}.json}
mkdir -p "$(dirname "$REPORT")"
echo "+ $PY $HERE/run_pipeline.py ${ARGS[*]} --json $REPORT $*" >&2
cd "$ROOT"
exec $PY "$HERE/run_pipeline.py" "${ARGS[@]}" --json "$REPORT" "$@"
