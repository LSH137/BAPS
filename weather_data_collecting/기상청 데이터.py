from selenium import webdriver 
from selenium.webdriver.support.ui import Select
import time 
url = 'http://taas.koroad.or.kr/gis/mcm/mcl/initMap.do?menuId=GIS_GMP_STS_RSN'
#들어가고자 하는 웹사이트 주소 
driver = webdriver.Chrome('./chromedriver')
#웹브라우저 (*https://chromedriver.chromium.org/downloads에서 exe 다운받은 뒤 파이썬 파일이 있는 폴더에 저장) 
driver.get(url) 
print(driver.window_handles)
main = driver.window_handles
'''
driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div/div[2]/div/div[2]/div/div[1]/div[2]/div[1]/ul/li[2]/input').click()
driver.find_element_by_xpath('//*[@id="ptsRafCh1AccidentContent"]/li[3]/input').click()
driver.find_element_by_xpath('//*[@id="ptsRafCh1AccidentContent"]/li[4]/input').click()
#작은 따옴표 안에 xpath를 넣으면 자동으로 클릭한다.
'''
select_fr = Select(driver.find_element_by_id("ptsRafYearStart"))
#html 파일에서 아이디를 이용해 객체 찾기 (개발자모드 f12)
select_fr.select_by_index(3)
#객체 값 선택하기
driver.find_element_by_class_name('btn-search').click()
#클래스 이름을 이용해 버튼을 찾아서 클릭하기
time.sleep(5) 
#조회되기까지 대기
driver.find_element_by_xpath('//*[@id="regionAccidentFind"]/div[3]/div[2]/p/a').click()
driver.switch_to_window(driver.window_handles[1])
#새 팝업창으로 이동
html =driver.page_source
#팝업창의 소스코드 저장
driver.close()
#팝업창 닫기
print(html)
driver.switch_to_window(driver.window_handles[0])
#처음 탭 열기
driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div/div[2]/div/div[2]/div/div[1]/div[2]/div[1]/ul/li[2]/input').click()