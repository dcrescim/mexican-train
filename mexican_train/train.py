import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from typing import Optional, List
from mexican_train.domino_types import Domino, random_string, is_double


class Train:
    """
    Representation of a train in the game.

    This class represents a train in the game. A train has a unique id,

    Attributes:
        id (str): The unique id of the train.
        dominoes (List[Domino]): The dominoes in the train.
        is_open (bool): Whether the train is open or not.
        player_id (Optional[str]): The unique id of the player who owns the train, if applicable.

    Raises:
        ValueError: If the train has no dominoes.
        ValueError: If the given domino cannot be added to the train.

    Examples:
        >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde")
        >>> train.add_domino((1, 2))
        >>> train.dominoes
        [(0, 0), (0, 1), (1, 1), (1, 2)]
        >>> train.add_dominoes([(2, 2), (2, 3)])
        >>> train.dominoes
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]
        >>> train.is_open_for_player("abcde")
        True
        >>> train.is_open_for_player("fghij")
        False
        >>> train.ends_in_double()
        False
        >>> train.get_end_value()
        3
    """

    def __init__(
        self,
        dominoes: List[Domino] = [],
        player_id: Optional[str] = None,
        is_open: bool = False,
        train_id: Optional[str] = None,
    ):
        """
        Initializes a train.

        Args:
            dominoes (List[Domino]): The dominoes in the train.
            player_id (Optional[str]): The unique id of the player who owns the train, if applicable.
            is_open (bool): Whether the train is open or not.
            train_id (Optional[str]): The unique id of the train. If None, then a random id will be generated.

        Raises:
            ValueError: If the train has no dominoes.

        Examples:
            >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde")
            >>> train.dominoes
            [(0, 0), (0, 1), (1, 1)]
            >>> train.player_id
            'abcde'
            >>> train.is_open
            False
            >>> train.id
            'xrioqj'
            >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde", is_open=True, train_id="abcde_player_train")
            >>> train.dominoes
            [(0, 0), (0, 1), (1, 1)]
            >>> train.player_id
            'abcde'
            >>> train.is_open
            True
            >>> train.id
            'abcde_player_train'
            >>> train = Train(dominoes=[])
            Traceback (most recent call last):
                ...
            ValueError: Train must have at least one domino
        """
        if train_id is not None:
            self.id = train_id
        else:
            self.id = random_string()
        if len(dominoes) == 0:
            raise ValueError("Train must have at least one domino")
        self.dominoes = dominoes
        self.is_open = is_open
        self.player_id = player_id

    def add_domino(self, new_domino: Domino) -> None:
        """
        Adds the given domino to the train.

        Args:
            new_domino (Domino): The domino to add to the train.

        Raises:
            ValueError: If the given domino cannot be added to the train.

        Examples:
            >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde")
            >>> train.add_domino((1, 2))
            >>> train.dominoes
            [(0, 0), (0, 1), (1, 1), (1, 2)]
            >>> train.add_domino((3, 3))
            Traceback (most recent call last):
                ...
            ValueError: Cannot add domino (3, 3) to train [(0, 0), (0, 1), (1, 1), (1, 2)]
        """
        last_domino = self.dominoes[-1]
        if last_domino[1] == new_domino[0]:
            self.dominoes.append(new_domino)
        elif last_domino[1] == new_domino[1]:
            self.dominoes.append((new_domino[1], new_domino[0]))
        else:
            raise ValueError(
                "Cannot add domino {} to train {}".format(new_domino, self.dominoes)
            )

    def add_dominoes(self, dominoes: List[Domino]) -> None:
        """
        Adds the given list of dominoes to the train.

        Args:
            dominoes (List[Domino]): The list of dominoes to add to the train.

        Raises:
            ValueError: If any of the given dominoes cannot be added to the train.

        Examples:
            >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde")
            >>> train.add_dominoes([(1, 2), (2, 3)])
            >>> train.dominoes
            [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
            >>> train.add_dominoes([(4, 5)])
            Traceback (most recent call last):
                ...
            ValueError: Cannot add domino (4, 5) to train [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
        """
        for domino in dominoes:
            self.add_domino(domino)

    def is_open_for_player(self, player_id: str) -> bool:
        """
        Returns whether the train is open for the given player.

        Args:
            player_id (str): The unique id of the player.

        Returns:
            bool: True if the train is open for the given player, False otherwise.

        Examples:
            >>> train = Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde")
            >>> train.is_open_for_player("abcde")
            True
            >>> train.is_open_for_player("fghij")
            False
            >>> train.is_open = True
            >>> train.is_open_for_player("abcde")
            True
            >>> train.is_open_for_player("fghij")
            True
            >>> train.is_open = False
            >>> train.player_id = None
            >>> train.is_open_for_player("abcde")
            True
            >>> train.is_open_for_player("fghij")
            True
        """
        return (
            (self.player_id is None) or (self.player_id == player_id) or (self.is_open)
        )

    def ends_in_double(self) -> bool:
        """
        Returns whether the train ends in a double.

        Returns:
            bool: True if the train ends in a double, False otherwise.
        """
        if len(self.dominoes) == 0:
            return False
        return is_double(self.dominoes[-1])

    def get_end_value(self) -> int:
        """
        Returns the value at the end of the train.

        Returns:
            int: The value at the end of the train.
        """
        if len(self.dominoes) == 0:
            raise ValueError("Train has no dominoes")
        return self.dominoes[-1][1]

    def __str__(self):
        """
        A string representation of the train that can be printed to the console

        Returns:
            str: A string representation of the train.
        """
        return str(
            {
                "id": self.id,
                "dominoes": self.dominoes,
                "is_open": self.is_open,
                "player_id": self.player_id,
            }
        )

    def __repr__(self):
        """
        A representation of the train that can be used for debugging
        or other purposes

        Returns:
            str: A representation of the train.
        """
        return str(
            {
                "id": self.id,
                "dominoes": self.dominoes,
                "is_open": self.is_open,
                "player_id": self.player_id,
            }
        )
