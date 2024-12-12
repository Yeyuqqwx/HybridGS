ONTHEGO_SCENES=("corner" "fountain" "mountain" "patio" "patio-high" "spot")
ROBUSTNERF_SCENES=("android" "statue" "yoda" "crab1" "crab2")
handle_scene() {
    for scene in "${@}"; do
        if [[ "$scene" == "$SCENE" ]]; then
            return 0
        fi
    done
    return 1
}

cd src

# for SCENE in "${ONTHEGO_SCENES[@]}"; do
for SCENE in "spot"; do
if handle_scene "${ONTHEGO_SCENES[@]}"; then
DATA_DIR= /path/to/dataset
CkPT_PATH=../output/onthego
CUDA_VISIBLE_DEVICES=0 python render_and_metric_3DGS.py --ckpt_path="${CkPT_PATH}/${SCENE}" \
    --source_path="${DATA_DIR}/${SCENE}" --resolution=1 \
    # --metric_only=True
elif handle_scene "${ROBUSTNERF_SCENES[@]}"; then
DATA_DIR= /path/to/dataset
CkPT_PATH=../output/robustnerf
CUDA_VISIBLE_DEVICES=0 python render_and_metric_gsplat.py --ckpt_path="${CkPT_PATH}/${SCENE}" \
    --data_dir="${DATA_DIR}/${SCENE}" --data_factor=8 \
    # --metric_only
fi
done