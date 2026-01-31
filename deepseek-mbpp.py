import math

import pandas as pd
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/djy/DEALRec-main/code/llama-7b-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/data/djy/DEALRec-main/code/llama-7b-hf", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# # tokenizer.eos_token_id is the id of <|EOT|> token
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


file_path = "Hall-Code2.xlsx"
# predict_list,label_list = extract_code_from_jsonl(file_path)
df1 = pd.read_excel(file_path, engine='openpyxl')
text_list = df1['instruction'].tolist()
code_list = df1['output'].tolist()
prompt0 = '''You are an expert Python test-driven developer.Create a new pytest test function that executes these missing lines/branches, always making sure that the new test is correct and indeed improves coverage.The test case madam coverage effect needs to satisfy the condation coverage criterion:such that the possible values of each condition in each judgement are satisfied at least once.Make sure the number of test cases is around 15.'''
prompt1="Please provide test cases for this code as per the above requirements"
for i in range(0, math.ceil(len(text_list) / 50)):
    answer1 = []
    answer2 = []
    if i == math.ceil(len(text_list) / 50) - 1:
        input = text_list[i * 50:]
        codes=code_list[i * 50:]
            # code = code_list[i * 50:]
    else:
        input = text_list[i * 50:(i + 1) * 50]
        codes=code_list[i * 50:(i + 1) * 50]
            # code = code_list[i * 50:(i + 1) * 50]
    for item1,item2 in zip(input,codes):
        prompt_step1 = prompt0 + prompt1 +".The code is" + str(item2)
        messages = [
            {'role': 'user', 'content': prompt_step1 }
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False,temperature=0.1, top_k=50, top_p=1,
                                 num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


        answer1.append(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

    data = {'text': input, 'code':codes,'gen_test': answer1}

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('./HC-temp-0.1-llama/hc' + str(i) + '.xlsx', index=False)
