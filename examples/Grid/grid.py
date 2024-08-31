import numpy as np 
from typing import NamedTuple, Tuple, List 

from reasoners.algorithm import MCTS, MCTSResult 
from reasoners.base import WorldModel, SearchConfig 

class GridState(NamedTuple):
    position: Tuple[int, int]
    grid: List[List[int]]

class GridAction(NamedTuple): 
    direction: str # up, down, left, right

class GridWorldModel(WorldModel[GridState, GridAction, None]):
    def __init__(self, grid: List[List[int]]): 
        self.grid = grid 
        self.height = len(grid)
        self.width = len(grid[0])

    def init_state(self) -> GridState:
        for i in range(self.height): 
            for j in range(self.width): 
                if self.grid[i][j] == 2:
                    return GridState((i,j), self.grid)
        raise ValueError("No initial position (2) found in the grid")
    
    def step(self, state: GridState, action: GridAction) -> Tuple[GridState, dict]: 
        x, y = state.position
        if action.direction == 'up': 
            new_x, new_y = x - 1, y
        elif action.direction == 'down': 
            new_x, new_y = x + 1, y 
        elif action.direction == 'left': 
            new_x, new_y = x, y - 1
        elif action.direction == 'right': 
            new_x, new_y = x, y + 1 
        else: 
            raise ValueError(f"Invalid action: {action}")
        
        # Check for valid position 
        if 0 <= new_x < self.height and 0 <= new_y < self.width and state.grid[new_x][new_y] != 1: 
            new_position = (new_x, new_y)
        else: 
            new_position = state.position

        new_state = GridState(new_position, state.grid)
        return new_state, {}
    
    def is_terminal(self, state: GridState) -> bool:
        x, y = state.position 
        return state.grid[x][y] == 3 
    
class GridSearchConfig(SearchConfig[GridState, GridAction, None]): 
    def __init__(self, grid: List[List[int]]): 
        super().__init__()
        self.grid = grid

    def get_actions(self, state: GridState) -> List[GridAction]:
        return[GridAction('up'), GridAction('down'), GridAction('left'), GridAction('right')]
    
    def reward(self, state: GridState, action: GridAction, **kwargs) -> Tuple[float, dict]: 
        new_state, _ = GridWorldModel(self.grid).step(state, action)

        if new_state.position == state.position: 
            return -0.1, {} # invalid move 
        elif self.grid[new_state.position[0]][new_state.position[1]] == 3: 
            return 1.0, {} # good move 
        else: 
            return -0.01, {} # small penalty for each step to encourage shorter path
        
class MCTSGridWrapper: 
    def __init__(self, grid: List[List[int]], n_iterations: int = 1000, exploration_weight: float = 1.0) -> None:
        self.grid = grid
        self.world_model = GridWorldModel(grid)
        self.search_config = GridSearchConfig(grid)
        self.search_algo = MCTS(
            n_iters= n_iterations, 
            w_exp=exploration_weight, 
            cum_reward=sum, 
            calc_q=np.mean, 
            simulate_strategy='random', 
            output_strategy='max_reward', 
            depth_limit=len(grid) * len(grid[0])
        )
        
    def __call__(self) -> MCTSResult: 
        return self.search_algo(self.world_model, self.search_config)
    
    @staticmethod
    def print_path(result: MCTSResult): 
        if result.trace is None: 
            print("No valid path found")
            return
        
        states, actions = result.trace
        print("Path found: ")
        for i, (state, action) in enumerate(zip(states, actions)): 
            print(f"Step{i}: Position {state.position}, Action: {action.direction}")

        print(f"Final position: {states[-1].position}")
        print(f"Cumulative reward: {result.cum_reward}")

if __name__ == "__main__": 
    # 0: Empty cell
    # 1: Blocked cell
    # 2: Initial position
    # 3: Exit (terminal state)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 3, 1, 2]
    ]

    mcts_wrapper = MCTSGridWrapper(grid, n_iterations=10000, exploration_weight=1.0)
    result = mcts_wrapper()

    MCTSGridWrapper.print_path(result)