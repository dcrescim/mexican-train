import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mexican_train.domino_types import GameLogEntry
from mexican_train.player import Player
from mexican_train.board import Board
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any


# Mexican Train Bot is a stub class that all player agents should inherit from
# it has a play method that takes in a player, board, is_first, and piece_counts


class MexicanTrainBot(ABC):
    """
    Representation of a Mexican Train bot.

    This class represents a Mexican Train bot. All player agents should
    inherit from this class and implement the play method.

    Attributes:
        name (str): The name of the bot.
    """

    def __init__(self, name: str):
        """
        Initializes a Mexican Train bot.

        """

        self.name = name

    def reset_attributes(self) -> None:
        """
        Resets the attributes of the bot before each game.

        Does not have to do anything, but can be overriden in child classes.
        """
        return

    @abstractmethod
    def play(
        self,
        player: Player,
        board: Board,
        is_first: bool,
        piece_counts: List[Tuple[str, int]],
        game_log: List[GameLogEntry],
    ) -> Optional[Any]:
        """
        The method to choose a `Move` to play. All player agents must implement
        this method.
        """
        pass
