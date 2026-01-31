import math

import pandas as pd
#from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/djy/step2/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/data/djy/step2/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# # tokenizer.eos_token_id is the id of <|EOT|> token
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

file_path = "/mnt/data/djy/step2/deepseek-wizard/HC-mcdc/merged-HC-mcdc.xlsx"
# predict_list,label_list = extract_code_from_jsonl(file_path)
df1 = pd.read_excel(file_path, engine='openpyxl')

    # 查看文件内容（可选）
    # print(df.head())  # 打印前几行，查看数据结构

    # 获取某一列的值
label_list = df1['code'].tolist()
test_list = df1['gen_test'].tolist()
#file_path2 = '../../data/MBPP/mbpp_out.xlsx'  # 替换为文件路径
#df = pd.read_excel(file_path2, engine='openpyxl')
#
#    # 查看文件内容（可选）
#    # print(df.head())  # 打印前几行，查看数据结构
#
#    # 获取某一列的值
#test_list = df['test'].tolist()
    # test_list=test_list[:50]
prompt0 = '''You are an expert Python  developer and an expert Python test-driven developer.. '''
prompta = '''What is the output of the following test inputs used with the code provided below, please give your answer.Please format the output strictly as follows
    {output:}'''
promptb='''Please compare the output of the original test case, if the input of the original test case and the output of the original test case are at least one of the same after running the provided program, then return true, otherwise answer false.Please format the output strictly as The answer you give can only be true or false, nothing more! '''
for i in range(0, math.ceil(len(label_list) / 50)):
    answer1 = []
    answer2 = []
    if i == math.ceil(len(label_list) / 50) - 1:
       put = label_list[i * 50:]
       test = test_list[i * 50:]
    else:
       put = label_list[i * 50:(i + 1) * 50]
       test = test_list[i * 50:(i + 1) * 50]
    for item1,item2 in zip(put,test):

        prompt_step1 = prompt0 + prompta + "The test inputs  is" + str(item2)+ ".The code is" + str(item1)
#        prompt_step1 = prompt0 + prompt1 +".The code is" + str(item2)
        messages = [
            {'role': 'user', 'content': prompt_step1 }
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95,
                                 num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


        answer1.append(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        prompt_step2 = prompt0 + promptb + "The test is" + str(item2) + ".The output after running the program is" + str(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        messages2 = [
            {'role': 'user', 'content': prompt_step2 }
        ]
        inputs2 = tokenizer.apply_chat_template(messages2, add_generation_prompt=True, return_tensors="pt").to(
            model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs2 = model.generate(inputs2, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95,
                                 num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs2[0][len(inputs2[0]):], skip_special_tokens=True))
        answer2.append(tokenizer.decode(outputs2[0][len(inputs2[0]):], skip_special_tokens=True))
    data = {'code': put, 'test': test, 'answer1': answer1,'answer2': answer2}

        # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

        # 将 DataFrame 写入 Excel 文件
    df.to_excel('/mnt/data/djy/step2/deepseek-wizard/passk/HC-mcdc/hc' + str(i) + '.xlsx', index=False)