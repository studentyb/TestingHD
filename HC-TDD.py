import math

import pandas as pd
#from modelscope import AutoTokenizer, AutoModelForCausalLM
#from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("/mnt/data/djy/step2/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/data/djy/step2/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# # tokenizer.eos_token_id is the id of <|EOT|> token
# outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
# print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))


file_path = "Hall-Code.xlsx"
# predict_list,label_list = extract_code_from_jsonl(file_path)
df1 = pd.read_excel(file_path, engine='openpyxl')
text_list = df1['instruction'].tolist()
code_list = df1['output'].tolist()
#text_list = df1['text'].tolist()
#code_list = df1['code'].tolist()
prompt0 = '''You are an expert Python test-driven developer.Creating  new pytest test function based on the user's needs always ensures that the new test is correct and actually improves coverage. Test cases should be created according to the boundary value analysis method of black box testing, where a boundary value is some specific situation that is slightly above the boundary or slightly below the boundary with respect to the input equivalence class and the output equivalence class.Make sure the number of test cases is around 15.'''
prompt1="Please provide test cases for this code as per the above requirements"
prompta='''Please extract the requirements for code generation in terms of language, algorithm, method name, input type, output type and other constraints based on the requirements given by the user.'''
prompt2="Creating new pytest test functions based on user requirements  and requirement limits extracted in the previous step ,always ensures that the new tests are correct and actually improve coverage. Test cases should be created according to the equivalence class division method of black-box testing. The equivalence class division method is a typical black-box testing method, which divides all possible input data (valid and invalid) to the programme into thousands of equivalence classes (the data in these equivalence classes are equivalent in terms of revealing the errors in the software), and selects representative data from each equivalence class as a test case to Achieve high test coverage with less data."
for i in range(54, math.ceil(len(text_list) / 50)):
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
        prompt_step1 = prompt0 + prompta + "The instruction  is" + item1
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
        prompt_step2 = prompt0 + prompt1 + prompt1  + "The instruction  is" + item1+ "The limits are" + tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
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

    data = {'text': input, 'code':codes,'limits': answer1,'gen_test': answer2}

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('./HC-TDD/HC' + str(i) + '.xlsx', index=False)
