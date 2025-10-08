set -ex
PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="aime24,aime25,amc23,math500,minerva_math,olympiadbench"
IFS="," read -ra BENCHMARK_ARRAY <<< "$DATA_NAME"
REGULAR_BENCHMARKS=()
SPECIAL_BENCHMARKS=()

for benchmark in "${BENCHMARK_ARRAY[@]}"; do
    if [[ "$benchmark" == "aime24" || "$benchmark" == "aime25" || "$benchmark" == "amc23" ]]; then
        SPECIAL_BENCHMARKS+=("$benchmark")
    else
        REGULAR_BENCHMARKS+=("$benchmark")
    fi
done

# Run regular benchmarks with n_sampling=1
if [ ${#REGULAR_BENCHMARKS[@]} -gt 0 ]; then
  REGULAR_BENCHMARKS_STR=$(IFS=,; echo "${REGULAR_BENCHMARKS[*]}")
  TOKENIZERS_PARALLELISM=false \
  python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${REGULAR_BENCHMARKS} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample 1 \
    --max_tokens_per_call 3000 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --overwrite \
    --save_outputs \
    --apply_chat_template
fi

if [ ${#SPECIAL_BENCHMARKS[@]} -gt 0 ]; then
  SPECIAL_BENCHMARKS=$(IFS=,; echo "${SPECIAL_BENCHMARKS[*]}")
  TOKENIZERS_PARALLELISM=false \
  python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${SPECIAL_BENCHMARKS_STR} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample 1 \
    --max_tokens_per_call 3000 \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 16 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --overwrite \
    --save_outputs \
    --apply_chat_template
fi
