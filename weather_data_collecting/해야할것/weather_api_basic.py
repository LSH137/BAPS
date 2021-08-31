from urllib.request import urlopen 
from urllib.parse import urlencode, unquote, quote_plus
import urllib 
import requests 
import json 
import pandas as pd



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

#