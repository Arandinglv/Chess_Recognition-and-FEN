# 计算滚轮的像素位置

import cv2


def calculate(func, threshold, screen, icon):

    screen = cv2.imread(screen)
    icon = cv2.imread(icon)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    match = cv2.matchTemplate(screen, icon, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

    if max_val < threshold:
        print(f"{func}按钮threshold为{match}")
        raise ValueError("无法找到该图标")
    else:
        print(f"{func}匹配成功!")
        icon_height, icon_width = icon.shape
        icon_center_x = icon_width // 2 + max_loc[0]
        icon_center_y = icon_height // 2 + max_loc[1]
        print(f"按钮{func}中心点坐标(x, y)为({icon_center_x}, {icon_center_y})")

    return icon_center_x, icon_center_y


func = "bottom"
threshold = 0.5
screen_bottom = "scroll_bottom.png"
icon_bottom = "./icon/scroll_bar_bottom.png"
icon_bottom_x, icon_bottom_y = calculate(func, threshold, screen_bottom, icon_bottom)

func = 'top'
icon_top = "./icon/scroll_bar.png"
screen_top = "scroll_top.png"
icon_top_x, icon_top_y = calculate(func, threshold, screen_top, icon_top)

icon_gap_x = icon_top_x - icon_bottom_x
icon_gap_y = icon_top_y - icon_bottom_y
print(icon_gap_x, icon_gap_y)

