import pandas as pd

# 파일 경로 설정
Location = 'D:/2_PythonExample'
File = 'accidentInfoList.xlsx'

# 추출 행, 열 선언
Row = 18
Column = 4

# 추출 및 변환 코드
data_pd = pd.read_excel('{}/{}'.format(Location, File), 
                     header=None, index_col=None, names=None)
data_np = pd.DataFrame.to_numpy(data_pd)

# 출력
print(data_pd)
print(data_np[Row][Column])
