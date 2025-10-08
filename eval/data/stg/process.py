import json
import random

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 转换为新的格式
def convert_to_new_format(data):
    converted_data = []
    for item in data:
        problem = item['problem']
        solution = item['solution']
        
        # 随机决定选项顺序
        if random.choice([True, False]):
            answer_choices = "(A) true (B) false"
            answer = "B" if solution == "true" else "A"
        else:
            answer_choices = "(A) false (B) true"
            answer = "A" if solution == "true" else "B"
        
        # 构造新的问题格式
        new_problem = f"{problem}\nAnswer Choices: {answer_choices}"
        
        # 构造新的JSON对象
        new_item = {
            "id": item['id'],
            "problem": new_problem,
            "answer": answer
        }
        converted_data.append(new_item)
    return converted_data

# 将结果写入新的JSON文件
def write_to_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in data:
            # 每个JSON对象占一行，不添加逗号或外部括号
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')  # 换行分隔每个JSON对象

# 主程序
if __name__ == "__main__":
    input_file = "/home/szhang/works/jywu/code/online_hiar_grpo/hiar_grpo/simpleRL-reason-main2/simpleRL-reason-main/eval/data/stg/stg_ori.json"  # 输入文件路径
    output_file = "/home/szhang/works/jywu/code/online_hiar_grpo/hiar_grpo/simpleRL-reason-main2/simpleRL-reason-main/eval/data/stg/stg.json"  # 输出文件路径
    
    # 读取数据
    data = read_json_file(input_file)
    
    # 转换为新的格式
    converted_data = convert_to_new_format(data)
    
    # 写入新的JSON文件
    write_to_json_file(converted_data, output_file)
    
    print(f"转换完成，结果已保存到 {output_file}")