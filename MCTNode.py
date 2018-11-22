import math

from Board import *
from Trees import TreeNode
from Utils import softmax


class MCTSNode(TreeNode):

    def __init__(self, board: Board, parent=None, children=None, wins=0, visits=0, state="OPEN", c=2.0, policy=None):
        super().__init__(parent=parent, children=children)
        self.board = board
        self.wins = wins
        self.visits = visits
        self.state = state
        self.c = c
        self.policy = policy

    def get_win_rate(self):
        return self.wins / self.visits

    def get_uct_policy_value(self):
        return self.get_win_rate() + self.policy * math.sqrt(self.c * self.parent.visits) / (1 + self.visits)

    def get_uct_value(self):
        return self.get_win_rate() + math.sqrt(self.c * math.log(self.parent.visits) / self.visits)

    def get_rank_value(self, uct_func=get_uct_value):
        '''
        sort key for node in the following descending order
        3: indicates a winning node for the parent
        2: indicates an unexplored node
        -1 to 1: explored nodes with UCT values scaled from -1 to 1 using tanh
        0: indicates a drawn node
        -2: indicates a losing node for the parent
        :return:
        '''
        if (self.board.is_white_to_move() and self.state == "WHITE_WON") or \
                (not self.board.is_white_to_move() and self.state == "BLACK_WON"):
            return -3
        if (not self.board.is_white_to_move() and self.state == "WHITE_WON") or \
                (self.board.is_white_to_move() and self.state == "BLACK_WON"):
            return 3
        if self.state == "DRAW":
            return 0.0
        if self.visits == 0:
            return 2
        return math.tanh(uct_func(self))

    def is_terminal_node(self):
        return self.state != "OPEN" or self.board.is_game_over()

    def backpropagate(self, game_result):
        self.visits += 1
        self.wins += -game_result if self.board.is_white_to_move() else game_result
        if not self.is_root():
            self.parent.backpropagate(game_result)

    def expand(self):
        for m in self.board.get_moves():
            self.board.move(m)
            if self.board.is_game_over():
                game_result = self.board.get_game_result()
                if game_result == 1:
                    self.state = "WHITE_WON"
                    self.add_child(MCTSNode(Board(self.board), state="WHITE_WON"))
                    self.board.un_move()
                    return
                if game_result == -1:
                    self.state = "BLACK_WON"
                    self.add_child(MCTSNode(Board(self.board), state="BLACK_WON"))
                    self.board.un_move()
                    return
                if game_result == 0:
                    self.add_child(MCTSNode(Board(self.board), state="DRAW"))
            else:
                self.add_child(MCTSNode(Board(self.board)))
            self.board.un_move()

    def expand_with_policy(self, policy_map):
        moves = self.board.get_moves()
        np_moves = moves_to_numpy(moves)
        policy_map[np_moves == 0] = 0
        policy_map = policy_map / np.sum(policy_map)
        for m in moves:
            self.board.move(m)
            if self.board.is_game_over():
                game_result = self.board.get_game_result()
                if game_result == 1:
                    self.state = "WHITE_WON"
                    self.add_child(MCTSNode(Board(self.board), state="WHITE_WON"))
                    self.board.un_move()
                    return
                if game_result == -1:
                    self.state = "BLACK_WON"
                    self.add_child(MCTSNode(Board(self.board), state="BLACK_WON"))
                    self.board.un_move()
                    return
                if game_result == 0:
                    self.add_child(MCTSNode(Board(self.board), state="DRAW"))
            else:
                self.add_child(MCTSNode(Board(self.board), policy=policy_map[move_to_index(m)]))
            self.board.un_move()

    def to_numpy_training_data(self):
        input_board = self.board.to_numpy()
        p = np.zeros((9, 9))
        if self.is_terminal_node():
            state_map = {
                "WHITE_WON": 1,
                "BLACK_WON": -1,
                "DRAW": 0
            }
            for c in self.children:
                if c.state == self.state:
                    m = get_last_move(c)
                    p[move_to_index(m)] = 1.0
            p = p / np.sum(p)
            v = state_map[self.state]
        else:
            v_array = np.zeros((9, 9))
            for c in self.children:
                m = get_last_move(c)
                m_index = move_to_index(m)
                p[m_index] = c.visits
                v_array[m_index] = c.wins / c.visits

            p = p / np.sum(p)
            # v = np.sum(v_array * p)  # p weighted v
            v = v_array[np.unravel_index(np.argmax(p), p.shape)]  # max p
        return input_board, p, v


def get_last_move(node: MCTSNode):
    return node.board.move_list[node.board.plies() - 1]
