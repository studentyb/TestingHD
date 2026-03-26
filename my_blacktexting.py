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

# 打开data.json文件并逐行读取
text_list=[]
code_list=[]
answer=[]
import pandas as pd

# 读取 Excel 文件
excel_file = pd.ExcelFile('dataset/result.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1',nrows=10)

# 分别获取 id 和 description 列的数据
id = df['id'].tolist()
description = df['description'].tolist()
buggy=df['buggy'].tolist()
fixed=df['fixed'].tolist()

prompt1='''You are an expert Python test-driven developer. '''

prompt3="Creating new pytest test functions based on user requirements always ensures that the new tests are correct and actually improve coverage. Test cases should be created according to the equivalence class division method of black-box testing. The equivalence class division method is a typical black-box testing method, which divides all possible input data (valid and invalid) to the programme into thousands of equivalence classes (the data in these equivalence classes are equivalent in terms of revealing the errors in the software), and selects representative data from each equivalence class as a test case to Achieve high test coverage with less data."
prompt4='''Always send entire Python test scripts when proposing a
new test or correcting one you previously proposed.
Be sure to include assertions in the test that verify any
applicable postconditions.
Please also make VERY SURE to clean up after the test,
so as not to affect other tests; use ’pytest-mock’ if
appropriate.
Write as little top-level code as possible, and in
particular do not include any top-level code calling into
pytest.main or the test itself.'''
prompt5='''Respond ONLY with the Python code enclosed in backticks,
without any explanation.And give in your answer what the exact value of the MCDC coverage of the test case is.The format of the generated test cases needs to conform to the format called by the Coverage tool.
.Please follow the following format strictly for the output format:
[
{
id:test case 1,
intput: ,
output.   
},
{
id:test case 2, intput: , output: }, {
intput: , output: }
output.   
}
]

'''
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-d6QZdnO61LR3QOGgDKEqoZtBmvqdwoejfldGZjW54FPMRuyg"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"
code='''
def find_similar_elements(list1, list2):
    set1 = set(tuple(x) for x in list1)
    set2 = set(tuple(x) for x in list2)
    common_elements = set1 & set2
    return list(common_elements)


'''
text='''Write a function to find the similar elements from the given two tuple lists.The function name and parameter settings are as follows: def find_similar_elements(list1, list2)'''
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
        api_key="",
        http_client=httpx.Client(
            base_url="https://xiaoai.plus/v1",
            follow_redirects=True,
        ),
    )
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 10
    # print(f"prompt:{prompt}")
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content":prompt}
  ],
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                n = requested_completions
            )

            completions.extend([choice.message.content for choice in response.choices])
            print(f"chat_response:{completions}")
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')

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


def run_test_with_coverage(item1,item2):
    # 创建一个 coverage 实例
    cov = coverage.Coverage()

    # 开始覆盖率收集
    cov.start()

    # 模拟输入数据
    input_data = "113.9 125.2 88.8\n"

    # 捕获输出
    with patch('sys.stdin', StringIO(input_data)), patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        # 运行目标代码
        list1 = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        list2 = [(2, 3, 4), (5, 6, 7), (8, 9, 10)]

        find_similar_elements(list1, list2)
        # 获取输出结果
        output = mock_stdout.getvalue().strip()
        expected_output = [(2, 3, 4), (5, 6, 7)]

        # 断言测试结果
        # assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # 停止覆盖率收集
    cov.stop()
    cov.save()

    # 合并并生成控制台覆盖率报告
    cov.combine()
    cov.report()

    # 生成HTML格式的详细报告
    cov.html_report(directory='htmlcov')

    print("HTML report generated in 'htmlcov/index.html'")
    # 提取未覆盖的行号
    # print("\nUncovered lines in 'my_code.py':")
    # analysis = cov.analysis('my_code.py')
    # lines = ""
    # for line in analysis[1]:
    #     print(f"Line {line} is not covered.")
    # print(analysis[1])
    # lines = ",".join(
    #     ",".join(map(str, item)) if isinstance(item, set) else str(item)
    #     for item in analysis[1]
    # )
    # print(lines)
    # prompt2 = "The code below does not achievefull coverage: when tested, lines " + str(lines) + " do not execute."
    #
    # prompt = prompt1 + prompt3 + prompt4 + prompt5 + "The requirements of the code are" +item1
    # output12=call_chatgpt_old(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
    #                  max_tokens=512, echo=False, majority_at=None)
    # answer.append(output12)


# 运行测试并统计覆盖率
for item1,item2 ,id1 in zip(description,fixed,id):
    print(id1)
    run_test_with_coverage(item1,item2)
data = {'id':id,'text': description ,'buggy':buggy, 'test':answer,'fixed':fixed}

# 将数据转换为 DataFrame
# df = pd.DataFrame(data)
#
# # 将 DataFrame 写入 Excel 文件
# df.to_excel('.\TB_dengjialei.xlsx', index=False)


