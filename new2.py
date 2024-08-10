"""
正式版, 图标匹配后到下棋之前
必须在24寸显示屏上运行
"""

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

driver.set_window_size(1800, 1000)  # 固定住这个分辨率


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
def calculate_pixel_pos(icon_path, func, threshold=0.5):
    screenshot_path = screenshot(func, driver)
    web_image = cv2.imread(screenshot_path)
    icon_image = cv2.imread(icon_path)
    web_image_gray = cv2.cvtColor(web_image, cv2.COLOR_BGR2GRAY)
    icon_image_gray = cv2.cvtColor(icon_image, cv2.COLOR_BGR2GRAY)
    match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
        match_result)  # tuple[float, float, cv2.typing.Point, cv2.typing.Point]

    if max_val < threshold:
        print(f"{func}按钮threshold为{match_result}")
        raise ValueError("无法找到该图标")
    else:
        print(f"{func}匹配成功!")
        icon_height, icon_width = icon_image_gray.shape
        icon_center_x = icon_width // 2 + max_loc[0]
        icon_center_y = icon_height // 2 + max_loc[1]
        print(f"按钮{func}中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

        return icon_center_x, icon_center_y


def icon_match(driver, login_path, func, test=True):
    icon_center_x, icon_center_y = calculate_pixel_pos(login_path, func)
    action = ActionChains(driver)
    try:
        time.sleep(1)
        action.move_by_offset(icon_center_x, icon_center_y).double_click().perform()
        print("点击成功")
    except Exception as e:
        print(f"点击操作被其他元素遮挡: {e}")
    action.move_by_offset(-icon_center_x, -icon_center_y).perform()  # 鼠标复位, 这一步非常关键

    # 验证是否进入界面
    if test:
        screenshot_test_path = screenshot(func, driver, test)
        return screenshot_test_path


def icon_match_single_click(driver, icon_path, func, test=True):
    """
    :param driver:
    :param login_path:
    :param func:
    :return:
    """
    icon_center_x, icon_center_y = calculate_pixel_pos(icon_path, func)
    action = ActionChains(driver)
    try:
        time.sleep(1)
        action.move_by_offset(icon_center_x, icon_center_y).click().perform()
        print("点击成功")
    except Exception as e:
        print(f"点击操作被其他元素遮挡: {e}")
    action.move_by_offset(- icon_center_x, - icon_center_y).perform()  # 鼠标复位, 这一步非常关键

    # 验证是否进入界面
    if test:
        screenshot_test_path = screenshot(func, driver, test=True)
        return screenshot_test_path


# TODO: 做不到拖着滚动条向上滑动
def drag(driver, start_x, start_y):
    """
    这个是滚动条函数, 向上向下滚动需要依赖于鼠标直接移动到需要滚动条滚动到的位置, 然后长按
    :param driver:
    :param start_x:
    :param start_y:
    :return:
    """
    action = ActionChains(driver)
    # 移动到起始位置, 拖住
    action.move_by_offset(start_x, start_y).click_and_hold().perform()
    # 移动到最终位置, 放下 move by offset是指在x和y方向上的拖拽距离
    time.sleep(3)
    # action.move_by_offset(0, 50).release()
    # 鼠标复位
    action.move_by_offset(- start_x, - start_y).perform()


# 等级条 -- 配合截图的方式, pixel-wise像素级别的点击不同等级
def level(driver, level_x, level_y):
    action = ActionChains(driver)
    action.move_by_offset(level_x, level_y).click().perform()
    time.sleep(2)
    action.move_by_offset(- level_x, - level_y).perform()
    print("点击成功")


# 爬虫方式
def set_slider_value(driver, target_value):
    """
    :param driver:
    :param target_value 必须在 [1, 25] 范围内, 并且1和2都是难度为250的等级上:
    :return:
    """
    try:
        # 如果元素能够立刻显示在页面上而不需要等待时间加载, 可以直接用driver.find_element
        # 如果需要加载的话, EC.presence_of_element_located
        slider = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "slider-input"))
        )
    except Exception:
        print("滑动条元素加载超时。")
        print(driver.page_source)  # 打印页面源代码以便于调试
        return

    # 获取滑动条的最小值和最大值
    min_value = int(slider.get_attribute("min"))
    max_value = int(slider.get_attribute("max"))

    # 检查目标值是否在有效范围内
    if target_value < min_value or target_value > max_value:
        raise ValueError(f"Target value {target_value} is out of range ({min_value}-{max_value})")

    # 设置滑动条的值
    driver.execute_script("arguments[0].value = arguments[1];", slider, target_value - 1)
    # 必须减一

    # 触发
    driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", slider)

############################# 进入登陆界面 #############################
icon_path = "./icon/login.png"
time.sleep(2)
icon_match(driver, icon_path, "login-in")
time.sleep(10)
# 进入到登陆界面, 输入用户名和密码
username = "arandinglv@163.com"
password = "20120907BYyx--"

# 账号
username_field = WebDriverWait(driver, 1).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input[aria-label='Username or Email']"))
)
username_field.send_keys(username)

# 密码
password_field = WebDriverWait(driver, 1).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
)
password_field.send_keys(password)

# 点击登录按钮
login_button = WebDriverWait(driver, 1).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "button#login"))
)
login_button.click()
time.sleep(2)
print("============ lo0gin success =============")

# ##################### 进入人机模式 vs computer ########################

icon_vs_computer_path = "./icon/vs_computer_2.png"
icon_match(driver, icon_vs_computer_path, "in_vs_computer")

time.sleep(2)


# ##################### 进入人机模式 vs computer start ########################

icon_vs_computer_path = "./icon/vs_computer_kaishi.png"
icon_match(driver, icon_vs_computer_path, "in_vs_computer_start")

time.sleep(2)


# ##################### 进入人机模式 鼠标滚动 ########################
bottom_x, bottom_y = 1446, 700  # 直接到底部
drag(driver, bottom_x, bottom_y)
time.sleep(2)


# ##################### 进入人机模式 选择engine ########################
icon_engine = './icon/engine.png'
icon_match_single_click(driver, icon_engine, 'choose_engine', test=False)
time.sleep(2)


# ##################### 进入人机模式 进入engine ########################
icon_choose = './icon/choose.png'
icon_match_single_click(driver, icon_choose, 'choose', test=True)
time.sleep(2)


# ##################### 进入人机模式 挑选难度等级 ########################
"""
选择对战nan'du
"""
# ==============图像匹配方法=================
"""                                       
滚动条用图像的方法不够准确，最好还是爬虫       
难度等级条 y = 281
难度等级条 x
| 难度 | x |
| 250 | 1055 - 1060 |
| 400 | 1060 经常点不到 |
| 550 | 1065 - 1070 |
| 700 | 1080 |
"""
# level_x = 1062
# level_y = 281
# level(driver, level_x, level_y)
# =========================================

set_slider_value(driver, 21)
time.sleep(2)


# # ##################### 选择对战模式 ########################
# icon_mode_path = ' ./icon/challenge.png'
# icon_match(driver, icon_mode_path, 'mode', test=False)
# time.sleep(2)

# TODO: 这一块好像不太对啊? 为什么上面调用我的函数就会报错cv2的错, 但是这种不封装的代码就没事?

# 图标匹配进入登录界面


icon_mode_challenge_path = './icon/challenge.png'


def mode(driver, icon_mode_challenge_path):
    # 截图并读取图像
    screenshot_mode_path = "screenshot_mode.png"
    with open(screenshot_mode_path, 'wb') as file:
        file.write(driver.get_screenshot_as_png())
    web_image_gray = cv2.cvtColor(cv2.imread(screenshot_mode_path), cv2.COLOR_BGR2GRAY)
    icon_image_gray = cv2.cvtColor(cv2.imread(icon_mode_challenge_path), cv2.COLOR_BGR2GRAY)

    # 图像匹配
    match_result = cv2.matchTemplate(web_image_gray, icon_image_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
    threshold = 0.5
    if max_val < threshold:
        raise ValueError("无法找到该图标")
    icon_center = (icon_image_gray.shape[1] // 2 + max_loc[0], icon_image_gray.shape[0] // 2 + max_loc[1])
    print(f"按钮中心点坐标(x, y)为{icon_center}")

    # 点击按钮
    actions = ActionChains(driver)
    actions.move_by_offset(icon_center[0], icon_center[1]).click().move_by_offset(-icon_center[0],                                                                              -icon_center[1]).perform()
    time.sleep(2)

mode(driver, icon_mode_challenge_path)


# ##################### 选择对战模式 ########################
icon_play_chess_path = './icon/play_chess.png'
icon_match(driver, icon_play_chess_path, 'play_chess')
time.sleep(2)

