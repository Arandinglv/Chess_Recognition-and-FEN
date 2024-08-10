from recognize import Recognize


recognize = Recognize("../images/001.png", "../model/chess_piece_model1.pth")
recognize.load_model()
results = recognize.predict_square()


class FEN:
    def __init__(self, matrix, side, king_moved, rook_moved):
        """
        :param matrix: 识别出来的棋盘数组
        :param side: 轮到黑方还是白方, 默认白方(0)在下黑方(1)在上, 白方先走
        :param king_moved: 王是否移动过, {0: False, 1: False}, False表示没移动
        :param rook_moved: 车是否移动过,
        {0: {"a": False, "h": False}, 1: {"a": False, "h": False}}, a h分别代表列
        """
        self.matrix = matrix
        self.side = side
        self.king_moved = king_moved
        self.rook_moved = rook_moved

    def matrix_to_fen_part1(self):
        """
        :return: 生成的是第一部分的fen
        """
        self.label_map = {
            0: 'r', 1: 'n', 2: 'b', 3: 'q', 4: 'k', 5: 'p',  # 黑子
            6: 'R', 7: 'N', 8: 'B', 9: 'Q', 10: 'K', 11: 'P',  # 白字
            12: ' '  # 空格
        }

        fen1 = ""
        for row in range(8):
            row_temp = ""
            empty_count = 0
            for column in range(8):
                piece = self.matrix[row][column]
                if piece == 12:
                    empty_count +=1
                else:
                    if empty_count > 0:
                        row_temp += str(empty_count)
                        empty_piece = 0
                    row_temp += self.label_map[piece]

            if empty_count != 0:
                row_temp += str(empty_count)

            if row < 7:
                fen1 += row_temp + "/"
            else:
                fen1 += row_temp

        return fen1

    def matrix_to_fen_part2(self):

        if self.side == 0:
            fen2 = "w"
        elif self.side == 1:
            fen2 = "b"

        return fen2


    # ============================ 3. 王车易位 ============================
    def matrix_to_fen_part3(self):
        castling_rights = ""

        # 白方短易位
        if not self.king_moved[0] and not self.rook_moved[0]["h"]:
            if self.can_castle_short((7, 4), (7, 7)):
                castling_rights += "K"

        # 白方长易位
        if not self.king_moved[0] and not self.rook_moved[0]["a"]:
            if self.can_castle_long((7, 4), (7, 0)):
                castling_rights += "Q"

        # 黑方短易位
        if not self.king_moved[1] and not self.rook_moved[1]["h"]:
            if self.can_castle_short((0, 4), (0, 7)):
                castling_rights += "k"

        # 黑方长易位
        if not self.king_moved[1] and not self.rook_moved[1]["a"]:
            if self.can_castle_long((0, 4), (0, 0)):
                castling_rights += "q"

        if castling_rights == "":
            castling_rights = "-"

        return castling_rights

    def can_castle_short(self, king_pos, rook_pos):
        return (self.is_path_clear(king_pos, (king_pos[0], rook_pos[1] - 1)) and
                not self.is_under_attack(king_pos) and
                not self.is_under_attack((king_pos[0], rook_pos[1] - 1)))

    def can_castle_long(self, king_pos, rook_pos):
        return (self.is_path_clear(king_pos, (king_pos[0], rook_pos[1] + 1)) and
                not self.is_under_attack(king_pos) and
                not self.is_under_attack((king_pos[0], rook_pos[1] + 1)))

    def is_path_clear(self, start, end):
        # 检查路径上是否有棋子阻挡
        for i in range(min(start[1], end[1]) + 1, max(start[1], end[1])):
            if self.matrix[start[0]][i] != 12:
                return False
        return True

    def is_under_attack(self, square):
        row, col = square
        enemy_pieces = {0: ['r', 'n', 'b', 'q', 'k', 'p'], 1: ['R', 'N', 'B', 'Q', 'K', 'P']}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

        # Check for attacks by sliding pieces (rook, bishop, queen)
        for direction in directions:
            r, c = row + direction[0], col + direction[1]
            while 0 <= r < 8 and 0 <= c < 8:
                piece = self.label_map[self.matrix[r][c]]
                if piece != ' ':
                    if piece in enemy_pieces[self.side] and self.can_attack(piece, (r, c), square):
                        return True
                    break
                r += direction[0]
                c += direction[1]

        # Check for attacks by knights
        for move in knight_moves:
            r, c = row + move[0], col + move[1]
            if 0 <= r < 8 and 0 <= c < 8:
                piece = self.label_map[self.matrix[r][c]]
                if piece in enemy_pieces[self.side] and piece.lower() == 'n':
                    return True

        # Check for attacks by pawns
        pawn_dir = -1 if self.side == 0 else 1
        for d_col in [-1, 1]:
            r, c = row + pawn_dir, col + d_col
            if 0 <= r < 8 and 0 <= c < 8:
                piece = self.label_map[self.matrix[r][c]]
                if piece in enemy_pieces[self.side] and piece.lower() == 'p':
                    return True

        # Check for attacks by kings
        for direction in directions:
            r, c = row + direction[0], col + direction[1]
            if 0 <= r < 8 and 0 <= c < 8:
                piece = self.label_map[self.matrix[r][c]]
                if piece in enemy_pieces[self.side] and piece.lower() == 'k':
                    return True

        return False

    def can_attack(self, piece, start, target):
        s_row, s_col = start
        t_row, t_col = target

        if piece.lower() == 'r':  # Rook
            return s_row == t_row or s_col == t_col
        elif piece.lower() == 'n':  # Knight
            return (abs(s_row - t_row) == 2 and abs(s_col - t_col) == 1) or (
                        abs(s_row - t_row) == 1 and abs(s_col - t_col) == 2)
        elif piece.lower() == 'b':  # Bishop
            return abs(s_row - t_row) == abs(s_col - t_col)
        elif piece.lower() == 'q':  # Queen
            return s_row == t_row or s_col == t_col or abs(s_row - t_row) == abs(s_col - t_col)
        elif piece.lower() == 'k':  # King
            return max(abs(s_row - t_row), abs(s_col - t_col)) == 1
        elif piece.lower() == 'p':  # Pawn
            direction = -1 if piece == 'P' else 1
            return t_row - s_row == direction and abs(t_col - s_col) == 1

        return False









