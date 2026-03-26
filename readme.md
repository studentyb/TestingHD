# conda 环境
```shell
conda env create -f environment.yml
conda activate Dang
```
# 运行TDD

```shell
#设置apikey和输入文件
api_key='******'
# 读取Excel文件
file_path = 'dataset/result.xlsx'  # 替换为文件路径
```
运行
```shell     
python TDD.py
```


# 运行黑盒测试

修改my_blacktexting.py

```python
# 读取 输入文件
excel_file = pd.ExcelFile('dataset/result.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1',nrows=10)

# 设置apikey
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key="",
        http_client=httpx.Client(
            base_url="https://xiaoai.plus/v1",
            follow_redirects=True,
        ),
    )

```

运行
```shell     
python my_blacktexting.py
```

# 运行差分测试
```shell     
python chafen.py
```



