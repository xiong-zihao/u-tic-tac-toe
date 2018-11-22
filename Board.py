import numpy as np
import random as rand


class Board:
    """
    Representation of a Ultimate Tic Tac Toe board game
    """
    board_mask = 0b111111111

    def __init__(self, other=None):
        """
        Initializes a Ultimate Tic Tac Toe board game
        :param other:
        """
        if type(other) is Board:
            self.white = other.white[:]
            self.black = other.black[:]
            self.global_white = other.global_white
            self.global_black = other.global_black
            self.active_block = other.active_block[:]
            self.move_list = other.move_list[:]
        else:
            self.white = [0] * 9
            self.black = [0] * 9
            self.global_white = 0
            self.global_black = 0
            self.active_block = [-1]
            self.move_list = []

    def to_numpy(self):
        """
        Convert the board to a numpy tensor representation with 6 layers
        Layer 0: White BitBoard
        Layer 1: Black BitBoard
        Layer 2: Active BitBoard
        Layer 3: White Won Global BitBoard
        Layer 4: Black Won Global BitBoard
        Layer 5: Drawn Global BitBoard
        :return: A (6, 9, 9) numpy array
        """
        result = np.zeros((6, 9, 9), dtype=np.int8)
        for i in range(81):
            global_row = i // 9
            global_col = i % 9
            active_row = global_row // 3
            active_col = global_col // 3
            active_block = active_col + active_row * 3
            local_row = global_row - active_row * 3
            local_col = global_col - active_col * 3
            local_index = local_col + local_row * 3
            n = 1 << local_index
            if (self.white[active_block] & n) != 0:
                result[0, global_row, global_col] = 1
            if (self.black[active_block] & n) != 0:
                result[1, global_row, global_col] = 1
            if self.active_block[self.plies()] == active_block or self.active_block[self.plies()] == -1:
                result[2, global_row, global_col] = 1
            if self.global_white == (1 << active_block):
                result[3, global_row, global_col] = 1
            if self.global_black == (1 << active_block):
                result[4, global_row, global_col] = 1
            if Board.board_mask == (self.white[active_block] | self.black[active_block]):
                result[5, global_row, global_col] = 1
        return result

    def __eq__(self, other):
        return type(other) is Board and \
               self.white == other.white and \
               self.black == other.black and \
               self.active_block == other.active_block

    def __str__(self):
        template = """\
 {}|{}|{} | {}|{}|{} | {}|{}|{}
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 ------+-------+------
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 ------+-------+------
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
 {}|{}|{} | {}|{}|{} | {}|{}|{} 
         """

        pieces = []
        for i in range(81):
            global_row = i // 9
            global_col = i % 9
            active_row = global_row // 3
            active_col = global_col // 3
            active_block = active_col + active_row * 3
            local_row = global_row - active_row * 3
            local_col = global_col - active_col * 3
            local_index = local_col + local_row * 3
            n = 1 << local_index
            if (self.white[active_block] & n) != 0:
                pieces.append('X')
            elif (self.black[active_block] & n) != 0:
                pieces.append('O')
            else:
                pieces.append('_')

        return template.format(*pieces)

    def get_game_result(self):
        if not self.is_game_over():
            return
        return 1 if is_won(self.global_white) else -1 if is_won(self.global_black) else 0

    def get_moves(self):
        """
        Returns a list of possible moves in tuple form
        m[0] : index of the active block of the move
        m[1] : 3x3 BitBoard of the move in the active block
        :return: A list of tuple of moves
        """
        result = []
        active = self.active_block[self.plies()]
        if active == -1:
            for i in range(9):
                n = 1 << i
                if n & (self.global_white | self.global_black) != 0 or \
                        Board.board_mask ^ (self.white[i] | self.black[i]) == 0:
                    continue
                empty = Board.board_mask & ~(self.white[i] | self.black[i])
                while empty != 0:
                    lsb = empty & -empty
                    result.append((i, lsb))
                    empty ^= lsb
        else:
            empty = Board.board_mask & ~(self.white[active] | self.black[active])
            while empty != 0:
                lsb = empty & -empty
                result.append((active, lsb))
                empty ^= lsb
        return result

    def move(self, m):
        active = m[1].bit_length() - 1
        if self.is_white_to_move():
            self.white[m[0]] ^= m[1]
            if is_won(self.white[m[0]]):
                self.global_white ^= 1 << m[0]
        else:
            self.black[m[0]] ^= m[1]
            if is_won(self.black[m[0]]):
                self.global_black ^= 1 << m[0]

        if m[1] & (self.global_white | self.global_black) != 0 or \
                Board.board_mask ^ (self.white[active] | self.black[active]) == 0:
            active = -1
        self.move_list.append(m)
        self.active_block.append(active)

    def un_move(self):
        m = self.move_list.pop()
        if self.is_white_to_move():
            if is_won(self.white[m[0]]):
                self.global_white ^= 1 << m[0]
            self.white[m[0]] ^= m[1]
        else:
            if is_won(self.black[m[0]]):
                self.global_black ^= 1 << m[0]
            self.black[m[0]] ^= m[1]
        self.active_block.pop()

    def plies(self):
        return len(self.move_list)

    def is_game_over(self):
        return is_won(self.global_white) or is_won(self.global_black) or len(
            self.get_moves()) == 0 or self.plies() == 80

    def is_white_to_move(self):
        return self.plies() % 2 == 0


def is_won(board):
    row = (((((board & 0b1001001) << 1) & board) << 1) & board) != 0
    col = (((((board & 0b111) << 3) & board) << 3) & board) != 0
    diag = (board & 0b100010001) == 0b100010001 or (board & 0b1010100) == 0b1010100
    return row or col or diag


def moves_to_numpy(moves):
    """
    Converts a list of moves into a 9x9 BitBoard
    :param moves: A list of moves
    :return: A (9, 9) numpy array
    """
    result = np.zeros((9, 9), dtype=np.int8)
    for m in moves:
        global_row, global_col = move_to_index(m)
        result[global_row, global_col] = 1
    return result


def move_to_index(m):
    active = m[0]
    local = m[1].bit_length() - 1
    active_row = active // 3
    active_col = active % 3
    local_row = local // 3
    local_col = local % 3
    global_row = active_row * 3 + local_row
    global_col = active_col * 3 + local_col
    return global_row, global_col


def index_to_move(global_row, global_col):
    active_row = global_row // 3
    active_col = global_col // 3
    active_block = active_col + active_row * 3
    local_row = global_row - active_row * 3
    local_col = global_col - active_col * 3
    local_index = local_col + local_row * 3
    n = 1 << local_index
    return active_block, n


if __name__ == '__main__':
    b = Board()
    print(all([index_to_move(*move_to_index(m)) == m for m in b.get_moves()]))
    # while not b.is_game_over():
    #     print(b)
    #     moves = b.get_moves()
    #     b.move(moves[rand.randrange(len(moves))])
    # print(b)
    # print(b.get_game_result())
