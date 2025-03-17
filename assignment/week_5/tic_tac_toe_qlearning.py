import numpy as np
import random
import pickle

class TicTacToeQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # Bảng Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
    
    def get_state(self, board):
        """Chuyển bàn cờ thành chuỗi để làm key trong Q-table"""
        return str(board.reshape(9))

    def get_q_value(self, state, action):
        """Lấy giá trị Q từ bảng"""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, board):
        """Chọn hành động dựa trên Q-learning hoặc random"""
        state = self.get_state(board)
        actions = [i for i in range(9) if board.flatten()[i] == 0]

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)  # Chọn ngẫu nhiên (exploration)

        # Chọn hành động tốt nhất (exploitation)
        q_values = [self.get_q_value(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, board, action, reward, next_board):
        """Cập nhật Q-table theo phương trình Q-learning"""
        state = self.get_state(board)
        next_state = self.get_state(next_board)
        
        max_next_q = max([self.get_q_value(next_state, a) for a in range(9) if next_board.flatten()[a] == 0], default=0)

        self.q_table[(state, action)] = self.get_q_value(state, action) + \
            self.alpha * (reward + self.gamma * max_next_q - self.get_q_value(state, action))

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

# Huấn luyện agent chơi với chính nó
def train_agent(episodes=50000):
    agent = TicTacToeQLearning()
    
    for episode in range(episodes):
        board = np.zeros((3, 3), dtype=int)
        done = False
        turn = 1  # 1: X, -1: O
        history = []

        while not done:
            action = agent.choose_action(board)
            board.flat[action] = turn
            state = agent.get_state(board)
            
            # Kiểm tra thắng/thua/hòa
            reward = 0
            if check_win(board, turn):
                reward = 1
                done = True
            elif np.all(board != 0):  # Hòa
                reward = 0
                done = True
            
            history.append((state, action, reward))
            turn *= -1  # Đổi lượt
            
            # Cập nhật Q-table sau khi ván đấu kết thúc
            if done:
                for i, (s, a, r) in enumerate(reversed(history)):
                    agent.update_q_table(board, a, r, board)

    agent.save_q_table()
    print("Training completed!")

def check_win(board, player):
    """Kiểm tra xem player có thắng không"""
    for row in board:
        if np.all(row == player):
            return True
    for col in board.T:
        if np.all(col == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# Chạy huấn luyện
train_agent()
