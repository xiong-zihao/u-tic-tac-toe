from Board import Board


def alpha_beta_search(board, depth, alpha=-float('inf'), beta=float('inf'), side=None):
    if side is None:
        side = 1 if board.is_white_to_move() else -1
    if board.is_game_over():
        return side * board.get_game_result(), None
    if depth == 0:
        return side * 0, None  # Change to eval function
    max_eval = -float('inf')
    best_move = None
    for m in board.get_moves():
        board.move(m)
        e = -alpha_beta_search(board, depth - 1, -beta, -alpha, -side)[0]
        board.un_move()
        if e > max_eval:
            max_eval = e
            best_move = m
        alpha = max(alpha, max_eval)
        if alpha >= beta:
            break

    return max_eval, best_move
