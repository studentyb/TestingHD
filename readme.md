Conda Environment

conda env create -f environment.yml
conda activate Dang

Run TDD

# Set API key and input file
api_key='******'
# Read Excel file
file_path = 'dataset/result.xlsx'  # Replace with file path

Run

python TDD.py



# Run Black-box Testing

Modify my_blacktexting.py

python
Read input file

excel_file = pd.ExcelFile(‘dataset/result.xlsx’)
Get data from specified worksheet

df = excel_file.parse(‘Sheet1’,nrows=10)
Set API key

client = OpenAI(
    base_url='https://xiaoai.plus/v1' ,
    api_key="",
    http_client=httpx.Client(
        base_url="https://xiaoai.plus/v1" ,
        follow_redirects=True,
    ),
)


Run

shell
python my_blacktexting.py


# Run Differential Testing

shell
python chafen.py

