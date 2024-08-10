from recognize import Recognize


class Compare:
    def __init__(self, chess_type=0):
        """
        :param matrix1:
        :param matrix2:
        :param chess_type: 0为默认状态下的白子在下黑子在上, 1相反
        """
        self.matrix1 = None
        self.matrix2 = None
        self.chess_type = chess_type

    def compare(self, matrix1, matrix2):
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        row = {
            0: 'a', 1: 'b', 2: 'c', 3: 'd',
            4: 'e', 5: 'f', 6: 'g', 7: 'h'
        }
        changes = []
        for i in range(8):
            for j in range(8):
                if self.chess_type == 0:
                    row_name = 8 - i
                    column_name = row[j]
                elif self.chess_type == 1:
                    row_name = i
                    column_name = row[1 - j]

                # 第一张图这个棋子不为空, 第二张图为空, 那么这个位置是起始位置
                if self.matrix1[i][j] != 12 and self.matrix2[i][j] == 12:
                    pre_piece = self.matrix1[i][j]  # previous
                    pre_piece_pos = f"{column_name}{row_name}"

                # 第二张图这个棋子发生了变化(第一张图可能为空, 也可能不为空, 被吃掉了), 且不为空
                if self.matrix1[i][j] != self.matrix2[i][j] and self.matrix2[i][j] != 12:
                    sub_piece = self.matrix2[i][j]  # subsequent
                    sub_piece_pos = f"{column_name}{row_name}"

        if pre_piece_pos and sub_piece_pos:
            move_str = f"{pre_piece_pos} -> {sub_piece_pos}"
            changes.append(move_str)

        return changes

recognize = Recognize("chess_piece_model1.pth", num_classes=13, channels=1)
result1, _ = recognize.print_info("../images/011.png")
result2, _ = recognize.print_info("../images/012.png")
compare = Compare()
changes = compare.compare(result1, result2)
print(changes)

