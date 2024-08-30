import numpy as np
from typing import List, Tuple
from copy import deepcopy

from reasoners.algorithm import MCTS 
from reasoners.base import WorldModel, SearchConfig, Reasoner

class TicTacToeState:
    def __init__(self, board: np.ndarray, player: int):
        self.board = board
        self.player = player

    def __str__(self):
        return f"Player: {self.player}\n" + "\n".join([" ".join(row) for row in self.board])

    def __eq__(self, other):
        return np.array_equal(self.board, other.board) and self.player == other.player

    def __hash__(self):
        return hash(str(self.board) + str(self.player))

class TicTacToeAction:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __str__(self):
        return f"({self.row}, {self.col})"

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __repr__(self):
        return self.__str__()  # Use the same string representation for __repr__

    def __hash__(self):
        return hash((self.row, self.col))

class TicTacToeWorldModel(WorldModel[TicTacToeState, TicTacToeAction, None]):
    _singleton_state = None
    def init_state(self) -> TicTacToeState:
        if TicTacToeWorldModel._singleton_state is None:
            TicTacToeWorldModel._singleton_state = TicTacToeState(np.full((3,3), '.'), 1)
        return TicTacToeWorldModel._singleton_state
    
    def make_move(self, action: TicTacToeAction): 
        state = TicTacToeWorldModel._singleton_state
        if state.board[action.row, action.col] == '.':
            state.board[action.row, action.col] = 'X' if state.player == 1 else 'O'
            state.player = 3 - state.player
        else:
            raise ValueError("Invalid move: Cell is already occupied.")

    def step(self, state: TicTacToeState, action: TicTacToeAction) -> Tuple[TicTacToeState, dict]:
        new_state = deepcopy(state)
        new_state.board[action.row, action.col] = 'X' if state.player == 1 else 'O'
        new_state.player = 3 - state.player  # Switch player (1 -> 2, 2 -> 1)
        return new_state, {}

    def is_terminal(self, state: TicTacToeState) -> bool:
        return self.check_win(state) or np.all(state.board != '.')

    @staticmethod
    def check_win(state: TicTacToeState) -> bool:
        board = state.board
        for i in range(3):
            if np.all(board[i, :] == board[i, 0]) and board[i, 0] != '.':
                return True
            if np.all(board[:, i] == board[0, i]) and board[0, i] != '.':
                return True
        if np.all(np.diag(board) == board[0, 0]) and board[0, 0] != '.':
            return True
        if np.all(np.diag(np.fliplr(board)) == board[0, 2]) and board[0, 2] != '.':
            return True
        return False

class TicTacToeSearchConfig(SearchConfig[TicTacToeState, TicTacToeAction, None]):
    def get_actions(self, state: TicTacToeState) -> List[TicTacToeAction]:
        return [TicTacToeAction(row, col) for row in range(3) for col in range(3) if state.board[row, col] == '.']

    def reward(self, state: TicTacToeState, action: TicTacToeAction, **kwargs) -> Tuple[float, dict]:
        new_state, _ = TicTacToeWorldModel().step(state, action)
        if TicTacToeWorldModel.check_win(new_state):
            return (1.0 if new_state.player == 1 else -1.0), {}  # Reward for the player who made the move
        elif np.all(new_state.board != '.'):
            return 0.0, {}  # Draw
        return 0.0, {}  # Non-terminal state

        # # Evaluate board position
        # player_pieces = np.sum(new_state.board == ('X' if new_state.player == 1 else 'O'))
        # opponent_pieces = np.sum(new_state.board == ('O' if new_state.player == 1 else 'X'))

        # # Prefer center and corners
        # position_value = 0.1 if action.row == 1 and action.col == 1 else 0.05 if action.row % 2 == 0 and action.col % 2 == 0 else 0

        # # Evaluate based on number of pieces and position
        # position_score = (player_pieces - opponent_pieces) / 5 + position_value

        #return 0.0, {}

class TicTacToeGame:
    def __init__(self):
        self.world_model = TicTacToeWorldModel()
        self.search_config = TicTacToeSearchConfig()
        self.mcts = MCTS(n_iters=10000, depth_limit=100)
        self.reasoner = Reasoner(self.world_model, self.search_config, self.mcts)

    def print_board(self, state: TicTacToeState):
        print(state)

    def get_human_move(self, state: TicTacToeState) -> TicTacToeAction:
        while True:
            try:
                row, col = map(int, input("Enter your move (row col): ").split())
                if 0 <= row < 3 and 0 <= col < 3 and state.board[row, col] == '.':
                    return TicTacToeAction(row, col)
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter two numbers separated by a space.")

    def get_ai_move(self, state: TicTacToeState) -> TicTacToeAction:
        print("AI is thinking...")
        #print(state)
        result = self.reasoner(None)

        # Debugging information
        print("MCTS Result:")
        print(f"Terminal state: {result.terminal_state}")
        print(f"Cumulative reward: {result.cum_reward}")

        if not result.trace or len(result.trace) < 2 or not result.trace[1]:
            print("Warning: MCTS returned an invalid trace. Falling back to random move.")
            return np.random.choice(self.search_config.get_actions(state))
        
        # print('--naman--')
        for state in result.trace[0]:
            print(state)
        print(result.trace[1])
        return result.trace[1][0]  # Get the first action from the best trace

    def play(self):
        state = self.world_model.init_state()
        print("Welcome to Tic Tac Toe!")
        print("You are 'X', and the AI is 'O'.")

        while not self.world_model.is_terminal(state):
            self.print_board(state)

            if state.player == 1:  # Human's turn
                action = self.get_human_move(state)
                print(f"You played: {action}")
            else:  # AI's turn
                action = self.get_ai_move(state)
                print(f"AI played: {action}")
            print(f"action {action}")
            self.world_model.make_move(action)

        self.print_board(state)

        if TicTacToeWorldModel.check_win(state):
            winner = "X" if state.player == 2 else "O"
            print(f"{winner} wins!")
        else:
            print("It's a draw!")

if __name__ == "__main__":
    game = TicTacToeGame()
    game.play()
