"""
识别
chess_piece_model.pth是彩色三通道
chess_piece_model1.pth是灰度三通道
"""

import cv2
import torch
from read_contour import Board
from model import MyResNet50
import os


class Recognize:
    def __init__(self, model_path, num_classes=13, channels=1):
        self.image_path = None
        self.model_path = model_path
        self.model = None
        self.height = None
        self.weight = None
        self.label_map = {
            0: 'rook', 1: 'knight', 2: 'bishop', 3: 'queen', 4: 'king', 5: 'soldier',  # 黑棋
            6: 'ROOK', 7: 'KNIGHT', 8: 'BISHOP', 9: 'QUEEN', 10: 'KING', 11: 'SOLDIER',  # 白棋
            12: ' '  # 空格
        }
        self.squares = None
        self.channels = channels
        self.num_classes = num_classes
        self.load_model()


    def split_board(self):
        board = Board(self.image_path)
        board_image = board.board_image()
        height, weight = board_image.shape[:2]
        self.height = height
        self.weight = weight
        square_size = height // 8
        squares = [[None for _ in range(8)] for _ in range(8)]

        for i in range(8):
            y_start = i * square_size
            y_end = y_start + square_size
            for j in range(8):
                x_start = j * square_size
                x_end = x_start + square_size
                square = board_image[y_start:y_end, x_start:x_end]
                squares[i][j] = square

        self.squares = squares

        return squares

    def load_model(self):
        self.model = MyResNet50(num_classes=self.num_classes, channel=self.channels)
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()


    # 三通道灰度图
    def preprocess_image(self, image_square):
        """
        为了让图像符合模型输入而进行预处理
        :param image_square: 单个square或者一个图像path
        :param input_square_type: True代表是square, False代表image_path, 需要imread
        :return:
        """
        image = cv2.resize(image_square, (224, 224))
        if self.channels == 3:
            image = image.astype('fload32') / 255.0
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        elif self.channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
            image = image.astype('float32') / 255.0
            image = torch.tensor(image).unsqueeze(0)  # 增加一个维度 (1, height, width)
            image = image.unsqueeze(0)  # 增加一个批次维度 (1, 3, height, width)
        return image

    def predict_all(self, image_path):
        """
        这个是一次性程序, 只能用于预测一整个8*8棋盘
        """
        self.image_path = image_path
        squares = self.split_board()
        results = [[None for _ in range(8)] for _ in range(8)]
        with torch.no_grad():
            for i in range(8):
                for j in range(8):
                    square = self.preprocess_image(squares[i][j])
                    output = self.model(square)
                    _, predicted = torch.max(output, 1)
                    results[i][j] = predicted.item()  # 将索引转换为实际的类别标签

        self.results = results

        return results

    def predict(self, square, input_square_type=True):
        """
        这个可以用来预测单个棋子块
        :param square: self.squares[i][j], 也可以是一张图像的path
        :param input_square_type: True代表她是square, False代表是一个image_path, 需要cv2.imread
        :return: result
        """
        if not input_square_type:
            image = cv2.imread(square)
        with torch.no_grad():
            image = self.preprocess_image(image)
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            result = predicted.item()

        return result

    def predict_square(self, image_path):
        """
        调用self.predict预测单个棋子后, 通过循环实现整个棋盘的预测
        :return:
        """
        self.image_path = image_path
        squares = self.split_board()
        results = [[None for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                square = squares[i][j]
                result = self.predict(square)
                results[i][j] = result

        self.results = results

        return results

    def print_info(self, image_path):
        results = self.predict_all(image_path)
        print(f"棋盘数组为:{results}")
        mapped_chessboard = []

        for row in results:
            mapped_row = []
            for idx in row:
                mapped_row.append(self.label_map[idx])
            mapped_chessboard.append(mapped_row)
        print(f"具体棋盘为:{mapped_chessboard}")

        return results, mapped_chessboard


# recognize = Recognize("chess_piece_model1.pth", num_classes=13, channels=1)
# recognize.print_info("../images/010.png")

