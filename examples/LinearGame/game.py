
from typing import List, Tuple
from reasoners.algorithm import MCTS 
from reasoners.base import WorldModel, SearchConfig, Reasoner


class LinearGameState:
    def __init__(self, position: int, max_position: int):
        self.position = position
        self.max_position = max_position

    def __str__(self):
        return f"Position: {self.position} / {self.max_position}"

    def __eq__(self, other):
        return self.position == other.position and self.max_position == other.max_position

    def __hash__(self):
        return hash((self.position, self.max_position))

class LinearGameAction:
    def __init__(self, move: int):
        self.move = move

    def __str__(self):
        return f"Move {self.move}"

    def __eq__(self, other):
        return self.move == other.move

    def __hash__(self):
        return hash(self.move)

class LinearGameWorldModel(WorldModel[LinearGameState, LinearGameAction, None]):
    def init_state(self) -> LinearGameState:
        return LinearGameState(0, 10)  # Start at 0, goal is 10

    def step(self, state: LinearGameState, action: LinearGameAction) -> Tuple[LinearGameState, dict]:
        new_position = min(state.position + action.move, state.max_position)
        return LinearGameState(new_position, state.max_position), {}

    def is_terminal(self, state: LinearGameState) -> bool:
        return state.position == state.max_position

class LinearGameSearchConfig(SearchConfig[LinearGameState, LinearGameAction, None]):
    def get_actions(self, state: LinearGameState) -> List[LinearGameAction]:
        return [LinearGameAction(1), LinearGameAction(2), LinearGameAction(3)]  # Can move 1, 2, or 3 steps

    def reward(self, state: LinearGameState, action: LinearGameAction, **kwargs) -> Tuple[float, dict]:
        new_state, _ = LinearGameWorldModel().step(state, action)
        if new_state.position == new_state.max_position:
            return 1.0, {}  # Reached the goal
        return 0.0, {}  # Not at goal yet

class LinearGame:
    def __init__(self):
        self.world_model = LinearGameWorldModel()
        self.search_config = LinearGameSearchConfig()
        self.mcts = MCTS(n_iters=1000, depth_limit=10)
        self.reasoner = Reasoner(self.world_model, self.search_config, self.mcts)

    def play(self):
        state = self.world_model.init_state()
        print("Welcome to the Linear Game!")
        print("Try to reach position 10 in the fewest moves.")
        print("You can move 1, 2, or 3 steps at a time.")

        moves = 0
        while not self.world_model.is_terminal(state):
            print(f"\nCurrent {state}")
            
            result = self.reasoner(state, prompt=None)
            
            if not result.trace or not result.trace[1]:
                print("Error: MCTS returned an empty trace.")
                break

            action = result.trace[1][0]
            state, _ = self.world_model.step(state, action)
            moves += 1

            print(f"AI chose: {action}")
            print(f"MCTS stats: Reward={result.cum_reward:.2f}, Depth={len(result.trace[1])}")

        print(f"\nGame Over! You reached the goal in {moves} moves.")

if __name__ == "__main__":
    game = LinearGame()
    game.play()