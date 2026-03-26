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

from my_blacktexting import prompt1

api_key='sk-d6QZdnO61LR3QOGgDKEqoZtBmvqdwoejfldGZjW54FPMRuyg'
# 读取Excel文件
file_path = 'dataset/result.xlsx'  # 替换为文件路径
df = pd.read_excel(file_path, engine='openpyxl',nrows=10)

# 查看文件内容（可选）
# print(df.head())  # 打印前几行，查看数据结构

# 获取某一列的值
id = df['id'].tolist()
description = df['description'].tolist()
buggy = df['buggy'].tolist()
fixed = df['fixed'].tolist()

# 打印结果
# print(column_values)
prompt0 = '''You are an expert Python  developer and an expert Python test-driven developer.. '''

prompta = '''Please extract the requirements for code generation in terms of language, algorithm, method name, input type, output type and other constraints based on the requirements given by the user.Please format the output strictly as follows
{function name:,Input type:,Output Type:,Memory Limit:,ADDitional Requirement:}'''
prompt3 = "Creating new pytest test functions based on user requirements  and requirement limits extracted in the previous step ,always ensures that the new tests are correct and actually improve coverage. Test cases should be created according to the equivalence class division method of black-box testing. The equivalence class division method is a typical black-box testing method, which divides all possible input data (valid and invalid) to the programme into thousands of equivalence classes (the data in these equivalence classes are equivalent in terms of revealing the errors in the software), and selects representative data from each equivalence class as a test case to Achieve high test coverage with less data."
prompt4 = '''Always send entire Python test scripts when proposing a
new test or correcting one you previously proposed.
Be sure to include assertions in the test that verify any
applicable postconditions.
Please also make VERY SURE to clean up after the test,
so as not to affect other tests; use ’pytest-mock’ if
appropriate.
Write as little top-level code as possible, and in
particular do not include any top-level code calling into
pytest.main or the test itself.'''
prompt5 = '''Respond ONLY with the Python code enclosed in backticks,
without any explanation.And give in your answer what the exact value of the MCDC coverage of the test case is.The format of the generated test cases needs to conform to the format called by the Coverage tool.
.Please follow the following format strictly for the output format,not generate any explain:
[{id:test case 1,intput: ,output:  }]
If it is not generated strictly according to the above format, please re-generate it，Make sure the number of test cases is around 15
'''
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = api_key
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"


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
        base_url='https://xiaoai.plus/v1',
        api_key=api_key,
        http_client=httpx.Client(
            base_url="https://xiaoai.plus/v1",
            follow_redirects=True,
        ),
    )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10
    print(f"prompt:{prompt}")
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


# 设置openai库的API密钥
# openai.api_key = "sk-yT6rG5Jcb4AOi8UY762a1dA5AfAa4938B704C205015b65Ae"
# 将目标代码嵌入到这个函数中

def find_similar_elements(list1, list2):
    # Convert the tuples to sets to remove duplicates and make comparison easier
    set1 = set(tuple(x) for x in list1)
    set2 = set(tuple(x) for x in list2)

    # Find the intersection of the two sets
    common_elements = set1 & set2;
    set2

    # Convert the set back to a list and return it
    return list(common_elements)


# def run_test_with_coverage(item1,item2):
#
#     prompt2 = "The code below does not achievefull coverage: when tested, lines " + str(lines) + " do not execute."
#
#     prompt = prompt1 + prompt3 + prompt4 + prompt5 + "The requirements of the code are" +item1+item2
#     output12=call_chatgpt_old(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
#                      max_tokens=512, echo=False, majority_at=None)
#     answer.append(output12)
print(math.ceil(len(description) / 50))
print(len(description) / 50)
for i in range(0, math.ceil(len(description) / 50)):
    answer1 = []
    answer2 = []
    if i == math.ceil(len(description) / 50) - 1:
        results = description[i * 50:]
        id1 = id[i * 50:]
    else:
        results = description[i * 50:(i + 1) * 50]
        id1 = id[i * 50:(i + 1) * 50]
    for item in results:
        #提取需求
        prompt_step1 = prompt0 + prompta + "The instruction  is" + item
        output1 = call_chatgpt_old(prompt_step1, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                                   max_tokens=512, echo=False, majority_at=None)
        print('output1'+output1)
        answer1.append(output1)

        prompt_step2 = prompt0 + prompt3 + prompt4 + prompt5 + "The instruction  is" + item + "The limits are" + \
                       output1[0]
        output2 = call_chatgpt_old(prompt_step2, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                                   max_tokens=512, echo=False, majority_at=None)
        answer2.append(output2)
    # 运行测试并统计覆盖率
    # for item1,item2 in zip(text_list,code_list):
    #     run_test_with_coverage(item1,item2)
    data = {'id': id1, 'text': results, 'code': answer1, 'test': answer2}

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 Excel 文件
    df.to_excel('.\\result\TB_dengjialei' + str(i) + '.xlsx', index=False)
