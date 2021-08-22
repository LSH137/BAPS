import csv
 
f = open('경로', 'r')#csv 파일의 경로
rdr = csv.reader(f) #csv파일을 읽습니다.
for line in rdr:
    print(line[13]) #각 줄의 데이터를 리스트로 읽습니다.
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