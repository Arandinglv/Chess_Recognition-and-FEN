import sys
sys.path.append('../')
import func

import time
import cv2

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


driver_path = "D:/Software/ChromeandChromeDriver/chromedriver-win64/chromedriver.exe"
service = Service(driver_path)
driver = webdriver.Chrome(service=service)
driver.get('http://www.chess.com')  # URL 参数需要包含协议http或者https
time.sleep(2)  # 加载页面至完整



# =============== 图标匹配 ================

driver.set_window_size(1800, 1200)
# 图标匹配进入登录界面
screenshot_path = "screenshot_login_start.png"
screenshot = driver.get_screenshot_as_png()
with open(screenshot_path, 'wb') as file:
    file.write(screenshot)

login_start_path = 'login.png'
# 读取页面截图和图标截图
web_image = cv2.imread(screenshot_path)
icon_image = cv2.imread(login_start_path)
# 灰度然后对比
web_image_gray = cv2.cvtColor(web_image, cv2.COLOR_BGR2GRAY)
icon_image_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)  # tuple[float, float, cv2.typing.Point, cv2.typing.Point]
threshold = 0.5

if max_val < threshold:
    raise ValueError("无法找到该图标")
else:
    icon_top_left = max_loc
    icon_bottom_right = min_loc
    icon_height, icon_weight = icon_image_gray.shape
    icon_center_x = icon_weight // 2 + max_loc[0]
    icon_center_y = icon_height // 2 + max_loc[1]
    print(f"按钮中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

    # 点击按钮
    action = ActionChains(driver)
    action.move_by_offset(icon_center_x, icon_center_y).click().perform()
    time.sleep(3)

    # ============ 爬虫 ==============
    # 进入到登陆界面, 输入用户名和密码
    username = "arandinglv@163.com"
    password = "20120907BYyx--"

    # 之前写的代码是:
    # driver.find_element(By.CSS_SELECTOR, "Username or Email").send_keys(username)
    # 会报错. 这是为什么呢?
    # 等待用户名输入框加载并输入用户名
    username_field = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='Username or Email']"))
    )
    username_field.send_keys(username)

    # 等待密码输入框加载并输入密码
    password_field = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
    )
    password_field.send_keys(password)

    # 点击登录按钮
    login_button = WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "button#login"))
    )
    login_button.click()

    # 等待登录完成，视需要调整等待时间
    time.sleep(5)

    # ============= 图标匹配 =============== 进不去啊, 这是为什么?

    screenshot_vs_computer_path = "screenshot_vs_computer.png"
    screenshot_vs_computer = driver.get_screenshot_as_png()
    with open(screenshot_vs_computer_path, "wb") as file:
        file.write(screenshot_vs_computer)

    icon_vs_computer_path = "challenge.png"
    # 读取页面截图和图标截图
    web_image = cv2.imread(screenshot_vs_computer_path)
    icon_image = cv2.imread(icon_vs_computer_path)
    # 灰度然后对比
    web_image_gray = cv2.cvtColor(web_image, cv2.COLOR_BGR2GRAY)
    icon_image_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
    match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        match_result)  # tuple[float, float, cv2.typing.Point, cv2.typing.Point]
    threshold = 0.5
    if max_val < threshold:
        raise ValueError("无法找到该图标")
    else:
        icon_top_left = max_loc
        icon_bottom_right = min_loc
        icon_height, icon_weight = icon_image_gray.shape
        icon_center_x = icon_weight // 2 + max_loc[0]
        icon_center_y = icon_height // 2 + max_loc[1]
        print(f"按钮中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

        # 点击按钮
        action = ActionChains(driver)
        action.move_by_offset(icon_center_x, icon_center_y).click().perform()
        time.sleep(60)




