import csv
 
f = open('/Users/a00/Desktop/BAPS/BAPS/weather_data_collecting/결과.csv', 'r')#csv 파일의 경로
rdr = csv.reader(f) #csv파일을 읽습니다.
for line in rdr:
    print(line[0], line[3]) #각 줄의 데이터를 리스트로 읽습니다.
'''
line[0] : 사고 번호
line[1] : 사고 시각
line[2] : 사고 요일
line[3] : 사고 위치
line[4] : 사고 내용
line[5] : 사망자 수
line[6] : 중상자 수
line[7] : 경상자 수
line[8] : 부상신고자 수
line[9] : 사고 유형
line[10] : 법규 위반
line[11] : 노면 상태
line[12] : 기상 상태
line[13] : 도로 형태
...이하 csv 파일 참고
'''
f.close()

<<<<<<< Updated upstream:weather_data_collecting/result.py
# CSV 파일에서 사고 시각과 위치를 가져오는 역할 
=======
# CSV 파일에서 사고 일시와 위치를 가져오는 역할
>>>>>>> Stashed changes:weather_data_collecting/해야할것/result.py
