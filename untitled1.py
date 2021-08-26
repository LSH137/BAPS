'''
#좌표값을 입력하면 행정 주소를 반환하는 코드입니다. 카카오 REST API를 활용했습니다.
import requests
import json

url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?'
headers = {"Authorization" : "KakaoAK 여기에 API 입력" }
params = {'x':127.1086228 , 'y' : 37.4012191}

places = requests.get(url, params=params, headers=headers).json()['documents'][0]['address_name']
print(places)
'''

'''
import requests

url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?'
headers = {"Authorization" : "KakaoAK bb5f9c89204f31ab89b43e0c10edaac5" }
params = {'x':127.1086228 , 'y' : 37.4012191}

places = requests.get(url, params=params, headers=headers).json()['documents'][0]['address_name']
print(places)
'''

#인근의 도로 종류를 찾습니다. (키워드 : 터널, 교량, 교차로 등 다 가능)
import requests

url = 'https://dapi.kakao.com/v2/local/search/keyword.json?'
headers = {"Authorization" : "KakaoAK bb5f9c89204f31ab89b43e0c10edaac5" }
params = {'x':127.1086228 , 'y' : 37.4012191, 'radius':200, 'query':'교량'}
#x, y: 중심 좌표, radius : 탐색 반경, query : 검색어 
places = requests.get(url, params=params, headers=headers).json()['meta']['total_count']
print(places)






