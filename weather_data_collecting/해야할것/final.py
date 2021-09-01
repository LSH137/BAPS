from urllib.request import urlopen 
from urllib.parse import urlencode, unquote, quote_plus
import urllib 
import requests 
import json 
import pandas as pd
import csv

position = { 
    '속초' : 90,
    '북춘천' : 93,
    '철원' : 95,
    '동두천' : 98,
    '파주' : 99,
    '대관령' : 100,
    '춘천' : 101,
    '백령도' : 102,
    '북강릉' : 104,
    '강릉' : 105,
    '동해' : 106,
    '서울' : 108,
    '인천' : 112,
    '원주' : 114,
    '울릉도' : 115,
    '수원' : 119,
    '영월' : 121,
    '충주' : 127,
    '서산' : 129, 
    '울진' : 130,
    '청주' : 131,
    '대전' : 133,
    '추풍령' : 135,
    '안동' : 136,
    '상주' : 137, 
    '포항' : 138,
    '군산' : 140,
    '홍천' : 212,
    '태백' : 216,
    '정선군' : 217,
    '제천' : 221,
    '보은' : 226,
    '천안' : 232,
    '보령' : 235,
    '부여' : 236,
    '금산' : 238,
    '세종' : 239,
    '부안' : 243,
    '임실' : 244,
    '정읍' : 245,
    '남원' : 247,
    '장수' : 248,
    '고창군' : 251,
    '영광군' : 252,
    '김해시' : 253,
    '순창군' : 254,
    '북창원' : 255,
    '양산시' : 257,
    '보성군' : 258,
    '강진군' : 259,
    '장흥' : 260,
    '해남' : 261,
    '고흥' : 262,
    '의령군' : 263,
    '대구' : 143,
    '전주' : 146,
    '울산' : 152,
    '창원' : 155,
    '광주' : 156,
    '부산' : 159,
    '통영' : 162,
    '목포' : 165,
    '여수' : 168,
    '흑산도' : 169,
    '완도' : 170,
    '고창' : 172,
    '순천' : 174,
    '홍성' : 177,
    '제주' : 184,
    '고산' : 185,
    '성산' : 188,
    '서귀포' : 189,
    '진주' : 192,
    '강화' : 201,
    '양평' : 202,
    '이천' : 203,
    '인제' : 211,
    '함양군' : 264,
    '광양시' : 266,
    '진도군' : 268,
    '봉화' : 271,
    '영주' : 272,
    '문경' : 273,
    '청송군' : 276,
    '영덕' : 277,
    '의성' : 278,
    '구미' : 279,
    '영천' : 281,
    '경주시' : 283,
    '거창' : 284,
    '합천' : 285,
    '밀양' : 288,
    '산청' : 289,
    '거제' : 294,
    '남해' : 295
}

f = open('/Users/a00/Downloads/결과.csv', 'r')#csv 파일의 경로
rdr = csv.reader(f) #csv파일을 읽습니다.

for line in rdr:
    print(line[0], line[3]) #각 줄의 데이터를 리스트로 읽습니다.
    if line[3] in position:
        area = line[3]
''' 참고 하기
line[0] : 사고 번호--> 날짜를 출력하게 됨.
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



url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : 'CLc5ewH3OFHBoz8idVj7vwmnjozhMiLB98sIf2BHZdI0NE5BuMvqY4gJhDb%2FGjYGvmuDpJa1N5hpammDWwLYkA%3D%3D', #키를 넣어야 API를 사용가능
                               quote_plus('pageNo') : '1', 
                               quote_plus('numOfRows') : '10', 
                               quote_plus('dataType') : 'json', 
                               quote_plus('dataCd') : 'ASOS', 
                               quote_plus('dateCd') : 'DAY', 
                               quote_plus('startDt') : '20100101', #시작 날짜--> 사고 날짜
                               #quote_plus('startHh') : '01', #시작 시--> 사고 당일 사건 시간
                               quote_plus('endDt') : '20100101', #종료 날짜--> 같은 날짜
                               #quote_plus('endHh') : '02',  # 한 시가의 차이가 나야한다.
                               quote_plus('stnIds') : '211' })
req = urllib.request.Request(url + unquote(queryParams))
response_body = urlopen(req).read()
dic=json.loads(response_body)
print(dic)

print("지역명 "+dic['response']['body']['items']['item'][0]['stnNm']) #지역명
print("기온 "+dic['response']['body']['items']['item'][0]['minTa']) #기온
print("강수량 "+dic['response']['body']['items']['item'][0]['sumRn']) #강수량
print("풍속 "+dic['response']['body']['items']['item'][0]['avgWs']) #풍속
print("습도 "+dic['response']['body']['items']['item'][0]['avgRhm']) #습도
print("적설 "+dic['response']['body']['items']['item'][0]['ddMes']) #적설

f.close()

#====== 


# CSV 파일에서 사고 시각과 위치를 가져오는 역할