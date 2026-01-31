import pandas as pd
import coverage
from unittest.mock import patch
from io import StringIO
from math import log
from decimal import Decimal
import coverage
from openai import OpenAI
import httpx
import time
import openai
import requests
import json
import os
import json
import math
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-HyB3eM2RlMcHwNDqFqnLHKluqZ3PBG3n1WwXMyZAXryH6T6e"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"

# text = '''Write a function to find the similar elements from the given two tuple lists.The function name and parameter settings are as follows: def find_similar_elements(list1, list2)'''


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=9k2MhCOp0APLJUkY15GbNZ1i&client_secret=27nxYD9TvoWjyXAFw5uuB1MokEIHwmdM"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def call_chatgpt_old(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                     max_tokens=512, echo=False, majority_at=None):
    client = OpenAI(
        base_url="https://xiaoai.plus/v1",
        api_key="sk-chZa0kWFdt101zkNDUwAq0lnL6zJBCTVrMr1GwzOt1P1cLRu",
        http_client=httpx.Client(
            base_url="https://xiaoai.plus/v1",
            follow_redirects=True,
            timeout=120
        )
    )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10
    print(f"prompthello:{prompt}")

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=requested_completions
            )

            completions.extend([choice.message.content for choice in response.choices])
            print(f"chat_response:{completions}")
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i ** 2, 60))
    raise RuntimeError('Failed to call GPT API')


def find_similar_elements(list1, list2):
    # Convert the tuples to sets to remove duplicates and make comparison easier
    set1 = set(tuple(x) for x in list1)
    set2 = set(tuple(x) for x in list2)

    # Find the intersection of the two sets
    common_elements = set1 & set2;
    set2

    # Convert the set back to a list and return it
    return list(common_elements)

def extract_code_from_jsonl(file_path):
    predict_list = []
    label_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if 'predict' in data:
                    predict_list.append(data['predict'])
                if 'label' in data:
                    label_list.append(data['label'])
    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
    except json.JSONDecodeError:
        print(f"解析 JSON 时出错，文件 {file_path} 可能不是有效的 JSONL 文件。")
    return predict_list,label_list


file_path = "E:\代码生成\deepseek-wizard\deepseek-wizard\HC-temp-0.7\merged-HC-temp-0.7.xlsx"
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
        output1 = call_chatgpt_old(prompt_step1, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                                   max_tokens=512, echo=False, majority_at=None)
        print(output1)

        answer1.append(output1)


        # answer1.append(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        prompt_step2 = prompt0 + promptb + "The test is" + str(item2) + ".The output after running the program is" + str(output1)
        output2 = call_chatgpt_old(prompt_step2, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                                   max_tokens=512, echo=False, majority_at=None)
        print(output2)

        answer2.append(output2)
        # answer2.append(tokenizer.decode(outputs2[0][len(inputs2[0]):], skip_special_tokens=True))
    data = {'code': put, 'test': test, 'answer1': answer1,'answer2': answer2}

        # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

        # 将 DataFrame 写入 Excel 文件
    df.to_excel('E:\代码生成\deepseek-wizard\deepseek-wizard\passk\HC-temp-detect-0.\hc' + str(i) + '.xlsx', index=False)