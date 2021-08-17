from selenium import webdriver 
from selenium.webdriver.support.ui import Select
import time 
def saver():
    global driver
    driver.find_element_by_class_name("btn-search").click()
    time.sleep(20)
    driver.find_element_by_xpath('//*[@id="regionAccidentFind"]/div[3]/div[2]/p/a').click()
    driver.switch_to_window(driver.window_handles[1])
    driver.find_element_by_class_name("pop-btn04").click()
    driver.close()
    driver.switch_to_window(driver.window_handles[0])    
global driver
url = 'http://taas.koroad.or.kr/gis/mcm/mcl/initMap.do?menuId=GIS_GMP_STS_RSN'
driver = webdriver.Chrome('./chromedriver')
driver.get(url)   
driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div/div[2]/div/div[2]/div/div[1]/div[2]/div[1]/ul/li[2]/input').click()
driver.find_element_by_xpath('//*[@id="ptsRafCh1AccidentContent"]/li[3]/input').click()
#driver.find_element_by_xpath('//*[@id="ptsRafCh1AccidentContent"]/li[4]/input').click()
select_place = Select(driver.find_element_by_id("ptsRafSido"))
select_all_place = Select(driver.find_element_by_id("ptsRafSigungu"))
for i in [1,4,7,10,13]:
    select_term = Select(driver.find_element_by_id("ptsRafYearStart"))
    select_term.select_by_index(i)
    time.sleep(2) #년도 선택
    for j in range(0,16):         
        select_place.select_by_index(j)
        time.sleep(1)
        select_all_place.select_by_index(0)
        time.sleep(1)
        saver()
    if i != 13:
        select_place.select_by_index(16)
        time.sleep(1) #위치
        select_all_place.select_by_index(0)
        time.sleep(1) #전체 선택
        saver()    