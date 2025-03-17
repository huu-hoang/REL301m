def play_with_agent():
    agent = TicTacToeQLearning()
    agent.load_q_table()

    board = np.zeros((3, 3), dtype=int)
    turn = 1  # 1: X, -1: O

    while True:
        print(board)
        if turn == 1:  # Người chơi (X)
            action = int(input("Nhập vị trí (0-8): "))
        else:  # AI chơi (O)
            action = agent.choose_action(board)

        if board.flatten()[action] != 0:
            print("Ô đã có người đánh!")
            continue

        board.flat[action] = turn

        if check_win(board, turn):
            print(board)
            print("Người chơi thắng!" if turn == 1 else "AI thắng!")
            break
        elif np.all(board != 0):  # Hòa
            print("Hòa!")
            break

        turn *= -1  # Đổi lượt

play_with_agent()
