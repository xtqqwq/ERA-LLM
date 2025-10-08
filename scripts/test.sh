cd ./eval
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="shepherd" # qwen25-math-cot / chatml
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=${2:-"./results"}
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
