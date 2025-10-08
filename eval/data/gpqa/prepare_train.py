import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('/home/szhang/works/jywu/code/online_hiar_grpo/hiar_grpo/simpleRL-reason-main2/simpleRL-reason-main/eval/data/gpqa/valid.gpqa.parquet')

# 将 DataFrame 转换为 JSON 格式并保存到文件
df.to_json('/home/szhang/works/jywu/code/online_hiar_grpo/hiar_grpo/simpleRL-reason-main2/simpleRL-reason-main/eval/data/gpqa/gpqa.json', orient='records', lines=True)