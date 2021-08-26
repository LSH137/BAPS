from urllib.request import urlopen 
from urllib.parse import urlencode, unquote, quote_plus 
import urllib
import requests 
import json 
import pandas as pd

# url for request 
url = 'http://data.kma.go.kr/apiData/getData'

# parameter for request
params = '?' + urlencode({
    quote_plus("type"): "json",
    quote_plus("dataCd"): "ASOS",
    quote_plus("dateCd"): "HR",
    quote_plus("startDt"): "20200212",
    quote_plus("startHh"): "00",
    quote_plus("endDt"): "20200212",
    quote_plus("endHh") : "23",
    quote_plus("stnIds"): "108",
    quote_plus("schListCnt"): "500",
    quote_plus("pageIndex"): "1",
    quote_plus("apiKey"): "사용자 키 입력"
})

req = urllib.request.Request(url + unquote(params))

# 의존성 추가
# import ssl
# context = ssl._create_unverified_context()

response_body = urlopen(req, timeout=60).read() # get bytes data
data = json.loads(response_body) # convert bytes data to json data
print(data)

data[0]['status']
data[1]['msg']
data[2]['stnIds']
res = pd.DataFrame(data[3]['info'])
print(res)

# 출처: https://signing.tistory.com/15 [끄적거림]