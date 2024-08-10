"""
生成标准棋盘, 也就是直接从网页上截取棋盘
"""

import cv2
import numpy as np

class Board:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None

    def angle_cos(self, A, B, C):
        """
        计算cos值, 如果两个边的向量点积为0, 则为直角边
          A |\
            | \
            |  \
            |   \
          B |____\C
        """
        AB_x = B[0] - A[0]
        AB_y = B[1] - A[1]
        BC_x = C[0] - B[0]
        BC_y = C[1] - B[1]
        if AB_x * AB_y + BC_x * BC_y == 0:
            return True
        else:
            return False

    def square_check(self):
        image = cv2.imread(self.image_path)
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        contour, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contour, key=cv2.contourArea)

        rect = cv2.minAreaRect(max_contour)
        points = cv2.boxPoints(rect)
        points = np.int0(points)

        # 验证角度是否为90度
        for i in range(4):
            if self.angle_cos(points[i], points[(i + 1) % 4], points[(i + 2) % 4]):
                continue
        # 验证边是否都相等
        if points[2][0] - points[0][0] != points[2][1] - points[0][1]:
            return False

        return points


    def board_image(self):
        points = self.square_check()
        board_image = self.image[points[0][1]:points[2][1],
               points[0][0]:points[2][0]]   # 先 y 后 x (先 H 后 W)
        return board_image


# image_path = '../images/002.png'  # 更新为上传文件的路径
# image = cv2.imread(image_path)
# board = Board(image_path)
# board_image = board.board_image()
# cv2.imshow("board_image", board_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

