from Board import Board

import random
import time
import numpy as np

from MCTNode import MCTSNode, get_last_move
from Utils import generate_symmetries


def mst_search(root: MCTSNode, timeout=5, max_nodes=1e3):
    """
    Implements a Monte Carlo Tree Search
    :param root: The root MCTSNode to perform the search on
    :param timeout: Timeout duration in seconds
    :param max_nodes: Max nodes searched
    :return: The root MCTSNode after the search
    """
    start = time.time()
    nodes = 0
    while time.time() - start < timeout and nodes < max_nodes and not root.is_terminal_node():
        nodes += 1
        best_node = select_best_node(root)
        if best_node.is_terminal_node():
            if best_node.state == "DRAW":
                best_node.backpropagate(0)
                continue
            if best_node.state == "WHITE_WON":
                best_node.backpropagate(1)
                continue
            if best_node.state == "BLACK_WON":
                best_node.backpropagate(-1)
                continue
        else:
            best_node.expand()
            if best_node.is_terminal_node():
                if best_node.state == "WHITE_WON":
                    best_node.backpropagate(1)
                    continue
                if best_node.state == "BLACK_WON":
                    best_node.backpropagate(-1)
                    continue
            random_node = best_node.children[random.randrange(len(best_node.children))]
            result = stimulate(random_node.board)
            random_node.backpropagate(result)
    # print("Time:{:0.2f} Nodes:{} NPS:{:0.2f}".format(time.time() - start, nodes, nodes/(time.time() - start)))
    return root


def get_best_child(root):
    if root.is_terminal_node():
        children = sorted(root.children, key=lambda node: node.get_rank_value(), reverse=True)
        return children[0]
    else:
        children = sorted(root.children, key=lambda node: node.visits, reverse=True)
        return children[0]


def select_best_node(root):
    while not (root.is_leaf() or root.is_terminal_node()):
        children = sorted(root.children, key=lambda n: n.get_rank_value(), reverse=True)
        best_node = children[0]
        if all([child.is_terminal_node() for child in children]):
            root.state = best_node.state
            return root
        if best_node.is_terminal_node():
            if best_node.state == "DRAW":
                return best_node
            root.state = best_node.state
            return root
        root = best_node
    return root


def stimulate(board: Board) -> int:
    b = Board(board)
    while not b.is_game_over():
        moves = b.get_moves()
        b.move(moves[random.randrange(len(moves))])
    return b.get_game_result()


def pit(is_white):
    b = Board()
    m = MCTSNode(b)
    while not b.is_game_over():
        if is_white == b.is_white_to_move():
            n = MCTSNode(b)
            n = mst_search(n)
            b.move(get_last_move(n))
        else:
            if m.board != b and (not (m.children is None)):
                m = next(filter(lambda node: node.board == b, m.children))
                if m is None:
                    print('Error')
                    raise Exception('Error')
            m = mst_search(m)
            b.move(get_last_move(m))
            m.parent.children = None
            m.parent = None
    return b.get_game_result() if is_white else -b.get_game_result()


if __name__ == '__main__':
    # sides = [True] * 10 + [False] * 10
    # with Pool(4) as pool:
    #     results = pool.map(pit, sides)
    # print(results)
    b = Board()
    input_data = []
    output_p_data = []

    output_v_data = []
    while not b.is_game_over():
        print(b)
        n = MCTSNode(b)
        n = mst_search(n)
        training_data = n.to_numpy_training_data()
        input_data.extend(generate_symmetries(training_data[0], axes=(1, 2)))
        output_p_data.extend(generate_symmetries(training_data[1]))
        output_v_data.extend([training_data[2]] * 8)
        best_child = get_best_child(n)
        b.move(get_last_move(best_child))

    print(b)
    print(b.get_game_result())
    input_array = np.array(input_data)
    output_p_array = np.array(output_p_data)
    output_v_array = np.array(output_v_data)
    np.savez('bootstrap', input=input_array, output_p=output_p_array, output_v=output_v_array)
