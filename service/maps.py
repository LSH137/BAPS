import folium as g
import cv2
import time
from selenium import webdriver
webdriver_options = webdriver.ChromeOptions()
webdriver_options .add_argument('headless')
driver = webdriver.Chrome(options=webdriver_options)
theta = 10
def image_cropper(image):
    width, height, channel = image.shape
    mat = cv2.getRotationMatrix2D((width/2, height/2), theta, 1)
    img1 = cv2.warpAffine(image, mat, (width, height))
    return img1

for i in range(10):
    p = [37.509671+i*0.0001, 127.055517]
    g_map = g.Map(location=p, zoom_start=18)
    marker = g.Marker(p, popup='campus seven', icon =g.Icon(icon='car',prefix='fa',color='blue'))
    marker.add_to(g_map)
    g_map.save('a.html')
    driver.get('C:\\Users\\podo3\\Desktop\\creative\\a.html')
    time.sleep(1)
    driver.save_screenshot("screenshot.png")
    image = cv2.imread("screenshot.png")
    cv2.imshow("adsf", image)
    cv2.waitKey(25)
cv2.destroyAllWindows()
driver.close()


   
    
    