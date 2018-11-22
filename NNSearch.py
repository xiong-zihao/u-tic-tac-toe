from MCTNode import MCTSNode, get_last_move
from NeuralNetwork import NeuralNetwork
from Board import Board
from MCTSearch import mst_search
import time
import random


def nn_search(root: MCTSNode, nn: NeuralNetwork, timeout=1, max_nodes=1e3):
    """
    Implements a Monte Carlo Tree Search with neural network evaluation
    :param root: The root MCTSNode to perform the search on
    :param nn: The neural network for evaluation
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
            p, v = nn.predict(best_node.board)
            best_node.expand_with_policy(p[0])
            if best_node.is_terminal_node():
                if best_node.state == "WHITE_WON":
                    best_node.backpropagate(1)
                    continue
                if best_node.state == "BLACK_WON":
                    best_node.backpropagate(-1)
                    continue
            best_node.backpropagate(v)
            # next_node = best_node.children[random.randrange(len(best_node.children))]
            # result = stimulate(next_node.board)
            # next_node.backpropagate(result)
    time_taken = time.time() - start
    print("Time:{:0.2f} Nodes:{} NPS:{:0.2f}".format(time_taken,
                                                     nodes,
                                                     float('inf') if time_taken == 0 else nodes / (
                                                                 time.time() - start)))
    # if root.is_terminal_node():
    #     children = sorted(root.children, key=lambda node: node.get_rank_value(), reverse=True)
    #     return children[0]
    # else:
    #     children = sorted(root.children, key=lambda node: node.visits, reverse=True)
    #     return children[0]
    return root


def select_best_node(root):
    while not (root.is_leaf() or root.is_terminal_node()):
        children = sorted(root.children,
                          key=lambda n: n.get_rank_value(uct_func=MCTSNode.get_uct_policy_value),
                          reverse=True)
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


def sample_best_move(root):
    """
    Samples the best move from the tree
    :param root: The root of the search tree
    :return: The best move
    """
    if root.is_terminal_node():
        children = sorted(root.children, key=lambda node: node.get_rank_value(), reverse=True)
        return get_last_move(children[0])
    else:
        child = random.choices(root.children, [c.visits for c in root.children])
        return get_last_move(child[0])


if __name__ == '__main__':
    b = Board()
    nn = NeuralNetwork()
    while not b.is_game_over():
        print(b)
        n = MCTSNode(b)
        if b.is_white_to_move():
            n = nn_search(n, nn)
            b.move(sample_best_move(n))
        else:
            n = mst_search(n)
            b.move(get_last_move(n))
    print(b)
    print(b.get_game_result())
