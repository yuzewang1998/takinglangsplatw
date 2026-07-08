#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/quickstart_render_eval.sh \
    --root_path <download_root> \
    --scene_name <scene_name> \
    --scene_dir <path/to/PT/<scene_name>> \
    --case_name <CASE_NAME> \
    [--exp_name <exp_name>]

This script renders the released MALE-GS checkpoints into the train-split
renders_npy folders required by eval/evaluate_iou_loc_pt.py, then runs PT-OVS
evaluation.

Required layout:
  <download_root>/benchmark/label/<scene_name>/*.json
  <download_root>/autoencoder/ckpt/<CASE_NAME>/best_ckpt.pth
  <download_root>/output/<exp_name>/<CASE_NAME>_{1,2,3}/chkpnt30000.pth

Options:
  --root_path PATH        Folder containing benchmark/, autoencoder/, output/.
  --scene_name NAME       PT-OVS scene name, e.g. trevi_fountain.
  --scene_dir PATH        Full scene folder used by render.py; must contain
                          images and sparse/ or dense/ camera data.
  --case_name NAME        Model/eval case name; must match <CASE_NAME>_{1,2,3}.
  --exp_name NAME         Experiment folder under output/. Defaults to CASE_NAME.
  --json_folder PATH      Label folder. Defaults to root_path/benchmark/label/scene_name.
  --output_dir PATH       Eval output folder. Defaults to root_path/eval_result.
  --resolution VALUE      Render/eval resolution. Defaults to 2.
  --mask_thresh VALUE     Eval mask threshold. Defaults to 0.4.
  --render_fusion NAME    Render-time feature fusion. Defaults to aug_wUncertainly_TMAM.
  --eval_fusion NAME      Eval-time post fusion. Defaults to post_validMapLevel_avgImageWiseMaxValue|LocMax.
  --language_features_name NAME
                          Defaults to language_features_dim3_<CASE_NAME>.
  --eval-only             Do not run render.py; only validate existing renders and evaluate.
  --render-only           Render and validate, but do not run evaluation.
  --no-sky-filter         Do not pass --sky_filter to evaluation.
  -h, --help              Show this help.
EOF
}

ROOT_PATH=""
SCENE_NAME=""
SCENE_DIR=""
CASE_NAME=""
EXP_NAME=""
JSON_FOLDER=""
OUTPUT_DIR=""
RESOLUTION="2"
MASK_THRESH="0.4"
RENDER_FUSION="aug_wUncertainly_TMAM"
EVAL_FUSION="post_validMapLevel_avgImageWiseMaxValue|LocMax"
LANGUAGE_FEATURES_NAME=""
RUN_RENDER=1
RUN_EVAL=1
SKY_FILTER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_path) ROOT_PATH="$2"; shift 2 ;;
    --scene_name) SCENE_NAME="$2"; shift 2 ;;
    --scene_dir) SCENE_DIR="$2"; shift 2 ;;
    --case_name) CASE_NAME="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    --json_folder) JSON_FOLDER="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --mask_thresh) MASK_THRESH="$2"; shift 2 ;;
    --render_fusion) RENDER_FUSION="$2"; shift 2 ;;
    --eval_fusion) EVAL_FUSION="$2"; shift 2 ;;
    --language_features_name) LANGUAGE_FEATURES_NAME="$2"; shift 2 ;;
    --eval-only) RUN_RENDER=0; shift ;;
    --render-only) RUN_EVAL=0; shift ;;
    --no-sky-filter) SKY_FILTER=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$ROOT_PATH" || -z "$SCENE_NAME" || -z "$SCENE_DIR" || -z "$CASE_NAME" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 2
fi

EXP_NAME="${EXP_NAME:-$CASE_NAME}"
JSON_FOLDER="${JSON_FOLDER:-${ROOT_PATH}/benchmark/label/${SCENE_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_PATH}/eval_result}"
LANGUAGE_FEATURES_NAME="${LANGUAGE_FEATURES_NAME:-language_features_dim3_${CASE_NAME}}"
OUTPUT_EXP_DIR="${ROOT_PATH}/output/${EXP_NAME}"
AE_CKPT_DIR="${ROOT_PATH}/autoencoder/ckpt"
AE_CKPT="${AE_CKPT_DIR}/${CASE_NAME}/best_ckpt.pth"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${message}: ${path}" >&2
    exit 1
  fi
}

require_path "$ROOT_PATH" "download root"
require_path "$SCENE_DIR" "scene_dir"
require_path "$JSON_FOLDER" "benchmark label folder"
require_path "$OUTPUT_EXP_DIR" "output experiment folder"
require_path "$AE_CKPT" "autoencoder checkpoint"

for level in 1 2 3; do
  require_path "${OUTPUT_EXP_DIR}/${CASE_NAME}_${level}/chkpnt30000.pth" "level ${level} checkpoint"
done

if [[ "$RUN_RENDER" -eq 1 ]]; then
  for level in 1 2 3; do
    model_dir="${OUTPUT_EXP_DIR}/${CASE_NAME}_${level}"
    echo "[QuickStart] Rendering feature level ${level} -> ${model_dir}/train/ours_None/renders_npy"
    PYTHONPATH=. python render.py \
      -s "$SCENE_DIR" \
      -m "$model_dir" \
      --feature_level "$level" \
      --include_feature \
      --resolution "$RESOLUTION" \
      --language_features_name "$LANGUAGE_FEATURES_NAME" \
      --which_feature_fusion_func "$RENDER_FUSION" \
      --render_small_batch \
      --render_json_folder "$JSON_FOLDER" \
      --skip_feature_gt \
      --skip_test \
      --skip_eval
  done
fi

echo "[QuickStart] Validating rendered npy files"
python - "$OUTPUT_EXP_DIR" "$CASE_NAME" "$JSON_FOLDER" <<'PY'
import glob
import json
import os
import sys
import numpy as np

output_exp_dir, case_name, json_folder = sys.argv[1:]
label_ids = []
for js in sorted(glob.glob(os.path.join(json_folder, "*.json"))):
    with open(js, "r") as f:
        data = json.load(f)
    name = data.get("info", {}).get("name")
    if name:
        label_ids.append(os.path.splitext(name)[0])

for level in (1, 2, 3):
    render_dir = os.path.join(output_exp_dir, f"{case_name}_{level}", "train", "ours_None", "renders_npy")
    files = sorted(glob.glob(os.path.join(render_dir, "*.npy")))
    if not files:
        raise SystemExit(f"No renders_npy files found for level {level}: {render_dir}")
    arr = np.load(files[0], mmap_mode="r")
    channels = arr.shape[-1]
    print(f"level {level}: {len(files)} files, sample shape {arr.shape}")
    if channels != 12:
        raise SystemExit(
            f"Expected 12-channel MALE-GS renders for post-fusion eval, got {channels} channels in {files[0]}. "
            "This usually means the checkpoint/render fusion is vanilla/default instead of aug_wUncertainly_TMAM."
        )
    if label_ids:
        available = {os.path.splitext(os.path.basename(p))[0] for p in files}
        missing = [idx for idx in label_ids if idx not in available]
        if missing:
            preview = ", ".join(missing[:5])
            raise SystemExit(
                f"Level {level} is missing {len(missing)} labeled render(s), e.g. {preview}. "
                "Make sure --scene_dir matches the benchmark scene and render.py generated the train split."
            )
PY

if [[ "$RUN_EVAL" -eq 1 ]]; then
  eval_args=(
    --dataset_name "$CASE_NAME"
    --feat_dir "$OUTPUT_EXP_DIR"
    --ae_ckpt_dir "$AE_CKPT_DIR"
    --output_dir "$OUTPUT_DIR"
    --mask_thresh "$MASK_THRESH"
    --resolution "$RESOLUTION"
    --encoder_dims 256 128 64 32 3
    --decoder_dims 16 32 64 128 256 256 512
    --json_folder "$JSON_FOLDER"
    --which_feature_fusion_func "$EVAL_FUSION"
  )
  if [[ "$SKY_FILTER" -eq 1 ]]; then
    eval_args+=(--sky_filter)
  fi
  echo "[QuickStart] Running PT-OVS evaluation"
  PYTHONPATH=. python eval/evaluate_iou_loc_pt.py "${eval_args[@]}"
fi
