from urllib.request import urlopen 
from urllib.parse import urlencode, unquote, quote_plus 
import urllib 
import requests 
import json 
import pandas as pd


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

url = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
queryParams = '?' + urlencode({ quote_plus('ServiceKey') : 'ol01kPg%2BpHOWuQZmRQmSc%2FtoW07yZ%2BZMGDokNgR2Q2F7rsx8wmL7D3j1QSOKsyZm0Am%2F0ko6FHX%2FZye2sXyQAQ%3D%3D', 
                               quote_plus('pageNo') : '1', 
                               quote_plus('numOfRows') : '10', 
                               quote_plus('dataType') : 'json', 
                               quote_plus('dataCd') : 'ASOS', 
                               quote_plus('dateCd') : 'HR', 
                               quote_plus('startDt') : '20100101', #시작 날짜 
                               quote_plus('startHh') : '01', #시작 시각
                               quote_plus('endDt') : '20100101', #종료 날짜 
                               quote_plus('endHh') : '02',  # 종료 시각
                               quote_plus('stnIds') : '108' })# 지역번호
req = urllib.request.Request(url + unquote(queryParams))
response_body = urlopen(req).read()
dic=json.loads(response_body)
print(dic)

''''
어떤 지역에 몇 번의 지역번호를 가지고 있는지 딕셔너리로 구현 키 = 지역 이름, 벨류를 지역번호
들어온 텍스트에서 지역번호 찾아 입력 번호 검색핫 ㅜ 있게

날짜와 시간 데이터에서 위의 형식으로 바꿔준다.
입력할 수 있게 함
시작 시간도 하면 엑셀의 형태를 가져와 돌리면 기상데이터를 가져올 수 있다. 
'''