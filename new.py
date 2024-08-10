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

driver.set_window_size(1800, 1000)


def screenshot(func, driver, test=False):
    if test:
        screenshot_path = f"screenshot_{func}_test.png"
    else:
        screenshot_path = f"screenshot_{func}.png"

    screenshot = driver.get_screenshot_as_png()
    with open(screenshot_path, 'wb') as file:
        file.write(screenshot)

    return screenshot_path


# 计算图标中心点坐标
def calculate_pixel_pos(driver, screenshot_path, login_path, func):

    web_image = cv2.imread(screenshot_path)
    icon_image = cv2.imread(login_path)
    web_image_gray = cv2.cvtColor(web_image, cv2.COLOR_BGR2GRAY)
    icon_image_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
    match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        match_result)  # tuple[float, float, cv2.typing.Point, cv2.typing.Point]
    threshold = 0.5

    if max_val < threshold:
        raise ValueError("无法找到该图标")
    else:
        print(f"{func}匹配成功!")
        icon_height, icon_width = icon_image_gray.shape
        icon_center_x = icon_width // 2 + max_loc[0]
        icon_center_y = icon_height // 2 + max_loc[1]
        print(f"按钮中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

        return icon_center_x, icon_center_y


def icon_match(driver, screenshot_path, login_path, func):
    web_image = cv2.imread(screenshot_path)
    icon_image = cv2.imread(login_path)

    web_image_gray = cv2.cvtColor(web_image, cv2.COLOR_BGR2GRAY)
    icon_image_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
    match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        match_result)  # tuple[float, float, cv2.typing.Point, cv2.typing.Point]
    threshold = 0.5

    if max_val < threshold:
        raise ValueError("无法找到该图标")
    else:
        print(f"{func}匹配成功!")
        icon_height, icon_width = icon_image_gray.shape
        icon_center_x = icon_width // 2 + max_loc[0]
        icon_center_y = icon_height // 2 + max_loc[1]
        print(f"按钮中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

        # 点击按钮
        action = ActionChains(driver)
        # body_element = driver.find_element(By.TAG_NAME, 'body')
        # action.move_to_element_with_offset(body_element, 0, 0)
        try:
            # action.move_to_element_with_offset(driver.find_element(By.TAG_NAME, 'body'), -10000, -10000).perform()
            time.sleep(1)
            action.move_by_offset(icon_center_x, icon_center_y).double_click().perform()
            print("点击成功")
        except Exception as e:
            print(f"点击操作被其他元素遮挡: {e}")
        action.move_by_offset(-icon_center_x, -icon_center_y).perform()  # 鼠标复位, 这一步非常关键

        # 验证是否进入login界面
        screenshot_test_path = screenshot(func, driver, test=True)
        return screenshot_test_path


################# 进入登陆界面 #################
screenshot_login_path = screenshot('login', driver)
icon_path = "./icon/login.png"
icon_match(driver, screenshot_login_path, icon_path, "login-in")
# 进入到登陆界面, 输入用户名和密码
username = "arandinglv@163.com"
password = "20120907BYyx--"

# 账号
username_field = WebDriverWait(driver, 3).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='Username or Email']"))
)
username_field.send_keys(username)

# 密码
password_field = WebDriverWait(driver, 3).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
)
password_field.send_keys(password)

# 点击登录按钮
login_button = WebDriverWait(driver, 3).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "button#login"))
)
login_button.click()
time.sleep(3)
print("============ login success =============")


# ######### 进入人机模式 vs computer ############

screenshot_vs_computer_path = screenshot('vc_computer', driver)
icon_vs_computer_path = "./icon/vs_computer_2.png"
icon_match(driver, screenshot_vs_computer_path, icon_vs_computer_path, "in_vs_computer")

time.sleep(3)


# ######### 进入人机模式 vs computer start ############

screenshot_vs_computer_path = screenshot('vc_computer_start', driver)
icon_vs_computer_path = "./icon/vs_computer_kaishi.png"
icon_match(driver, screenshot_vs_computer_path, icon_vs_computer_path, "in_vs_computer_start")

time.sleep(3)


# ######### 进入人机模式 人像匹配 ############



