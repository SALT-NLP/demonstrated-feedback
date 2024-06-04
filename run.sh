conda activate ditto

export HF_TOKEN=""

benchmark="custom"

rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file configs/multi_gpu.yaml \
    scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
    --train_pkl=benchmarks/${benchmark}/processed/custom_train.pkl \
    --train_author_key=0 \
    --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto \
    --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl

python generate.py \
    --benchmark=$benchmark \
    --train_author_key=0

