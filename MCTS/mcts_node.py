
from typing import Generic, TypeVar, Optional, NamedTuple, Callable, Hashable
import itertools
import numpy as np

Node_info = TypeVar("Node_info")
Action = TypeVar("Action")
Example = TypeVar("Example")

class MCTSNode(Generic[Node_info, Action, Example]):
    
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state, node_info: Optional[Node_info], action: Optional[Action], parent: "Optional[MCTSNode]" = None,
                 score: float = 0., score_details=None,
                 is_terminal: bool = False, calc_q: Callable[[list[float]], float] = np.mean):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param score: an estimation of the score of the current state
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """

        self.id = next(MCTSNode.id_iter)
        if score_details is None:
            score_details = {}
        
        self.cum_rewards: list[float] = []

        self.state = state
        self.score = self.reward = score
        self.score_details = score_details

        self.is_terminal = is_terminal
        self.action = action
        self.node_info = node_info
        self.parent = parent
        self.children: 'Optional[list[MCTSNode]]' = None
        self.calc_q = calc_q
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    @property
    def Q(self) -> float:
        if self.state is None:
            return self.score
        else:
            if self.cum_rewards:
                return self.calc_q(self.cum_rewards)
            else:
                return 0
            
    def serialize(self):
        """
        Serializes the node and its children into a dictionary format.
        """
        return {
            'id': self.id,  # Unique identifier for the node
            'state': self.state,  # Make sure `state` is serializable
            'score': self.score,
            'score_details': self.score_details,  # Ensure details are serializable
            'is_terminal': self.is_terminal,
            'action': self.action,
            'node_info': self.node_info,  # Ensure node_info is serializable
            'parent_id': self.parent.id if self.parent else None,  # Store parent ID for reference
            'depth': self.depth,
            'children': [child.serialize() for child in (self.children or [])],  # Recursively serialize children
            'cum_rewards': self.cum_rewards,  # History of cumulative rewards
        }