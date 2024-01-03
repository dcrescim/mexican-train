import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from typing import List, Optional
from mexican_train.domino_types import Domino, random_string, is_double


class Player:
    """
    Representation of a player in the game.

    This class represents a player in the game. A player has a unique id and
    a list of dominoes in their hand.

    Attributes:
        id (str): The unique id of the player.
        dominoes (List[Domino]): The dominoes in the player's hand.
    """

    def __init__(self, dominoes: List[Domino] = [], player_id: Optional[str] = None):
        """
        Initializes a player.

        Args:
            dominoes (List[Domino]): The dominoes in the player's hand.
            player_id (Optional[str]): The unique id of the player. If None, then a random id will be generated.
        """
        if player_id is None:
            self.id = random_string(6)
        else:
            self.id = player_id
        self.dominoes = dominoes

    def pieces_left(self) -> int:
        """
        The number of dominoes left in the player's hand.

        Returns:
            int: The number of dominoes left in the player's hand.

        Examples:
            >>> player = Player(dominoes=[(0, 0), (0, 1), (1, 1)])
            >>> player.pieces_left
            3
            >>> player.remove_domino((0, 0))
            >>> player.pieces_left
            2
            >>> player.remove_dominoes([(0, 1), (1, 1)])
            >>> player.pieces_left
            0
        """
        return len(self.dominoes)

    def get_highest_double(self) -> Optional[Domino]:
        """
        Returns the highest double in the player's hand if one exists,
        otherwise returns None.

        Returns:
            Optional[Domino]: The highest double in the player's hand if one exists, otherwise None.

        Examples:
            >>> player = Player(dominoes=[(0, 0), (0, 1), (1, 1)])
            >>> player.get_highest_double()
            (1, 1)
            >>> player = Player(dominoes=[(0, 1), (0, 2), (1, 2)])
            >>> player.get_highest_double()
            None
        """
        highest_double = None
        for domino in self.dominoes:
            if is_double(domino):
                if highest_double is None or domino[0] > highest_double[0]:
                    highest_double = domino
        return highest_double

    def remove_domino(self, domino: Domino) -> None:
        """
        Removes the given domino from the player's hand. If the domino
        is not in the player's hand, raises a ValueError.

        Args:
            domino (Domino): The domino to remove from the player's hand.

        Raises:
            ValueError: If the domino is not in the player's hand.

        Examples:
            >>> player = Player(dominoes=[(0, 0), (0, 1), (1, 1)])
            >>> player.remove_domino((1, 0))
            >>> player.dominoes
            [(0, 0), (1, 1)]
            >>> player.remove_domino((1, 1))
            >>> player.dominoes
            [(0, 0)]
            >>> player.remove_domino((0, 2))
            Traceback (most recent call last):
                ...
            ValueError: Tried to remove domino that didn't exist
        """
        if domino in self.dominoes:
            self.dominoes.remove(domino)
            return
        if (domino[1], domino[0]) in self.dominoes:
            self.dominoes.remove((domino[1], domino[0]))
            return
        raise ValueError("Tried to remove domino that didn't exist")

    def remove_dominoes(self, dominoes: List[Domino]) -> None:
        """
        Removes the given list of dominoes from the player's hand.
        Will raise a ValueError if any of the dominoes are not in
        the player's hand, but will remove all dominoes up to the
        point of the error.

        Args:
            dominoes (List[Domino]): The list of dominoes to remove from the player's hand.

        Raises:
            ValueError: If any of the dominoes are not in the player's hand.

        Examples:
            >>> player = Player(dominoes=[(0, 0), (0, 1), (1, 1)])
            >>> player.remove_dominoes([(0, 0), (1, 1)])
            >>> player.dominoes
            [(0, 1)]
            >>> player.remove_dominoes([(0, 1), (1, 1)])
            Traceback (most recent call last):
                ...
            ValueError: Tried to remove domino that didn't exist
            >>> player.dominoes
            [(0, 1)]
        """
        for domino in dominoes:
            self.remove_domino(domino)

    def __str__(self):
        """
        A string representation of the player that can be printed to the console
        """
        return str({"id": self.id, "dominoes": self.dominoes})

    def __repr__(self):
        """
        A string representation of the player that can be printed to the console
        """
        return str({"id": self.id, "dominoes": self.dominoes})
