import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from typing import Tuple, Optional, List
from mexican_train.domino_types import Domino, Continuation
from mexican_train.train import Train
from mexican_train.player import Player
from mexican_train.move import Move


class Board:
    """
    Representation of the board in the game.

    This class represents the board in the game. The board has a list of
    trains and an engine (the domino in the center of the board that
    all trains must start with).

    Attributes:
        trains (List[Train]): The trains on the board.
        engine (Optional[Domino]): The engine in the center of the board.

    Examples:
        >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="fghij")], engine=(0, 0))
        >>> board.trains
        [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'fghij'}]
        >>> board.engine
        (0, 0)
        >>> board.contains_unfulfilled_double
        False
        >>> board.unfulfilled_double_value
        >>> board.unfulfilled_double_train_id
        >>> board.get_open_trains("abcde")
        [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}]
        >>> board.get_open_trains("fghij")
        [{'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'fghij'}]
        >>> board.open_train(Player(dominoes=[], player_id="abcde"))
        >>> board.trains
        [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': True, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'fghij'}]
        >>> board.close_train(Player(dominoes=[], player_id="abcde"))
        >>> board.trains
        [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'fghij'}]
    """

    def __init__(self, trains: List[Train] = [], engine: Optional[Domino] = None):
        """
        Initializes a board.

        Args:
            trains (List[Train]): The trains on the board.
            engine (Optional[Domino]): The engine in the center of the board.

        Examples:
            >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 4), (4, 2)], player_id="fghij")], engine=(0, 0))
            >>> board.trains
            [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 4), (4, 2)], 'is_open': False, 'player_id': 'fghij'}]
            >>> board.engine
            (0, 0)
        """
        self.trains = trains
        self.engine = engine

    @property
    def contains_unfulfilled_double(self) -> bool:
        """
        Whether the board contains an unfulfilled double.

        Returns:
            bool: True if the board contains an unfulfilled double, False otherwise.
        """
        for train in self.trains:
            if train.ends_in_double():
                return True
        return False

    @property
    def unfulfilled_double_value(self) -> Optional[int]:
        """
        The value of the unfulfilled double, if applicable.

        Returns:
            Optional[int]: The value of the unfulfilled double, if applicable.
        """
        for train in self.trains:
            if train.ends_in_double():
                return train.get_end_value()
        return None

    @property
    def unfulfilled_double_train_id(self) -> Optional[str]:
        """
        The id of the train with the unfulfilled double, if applicable.

        Returns:
            Optional[str]: The id of the train with the unfulfilled double, if applicable.
        """
        for train in self.trains:
            if train.ends_in_double():
                return train.id
        return None

    def get_open_trains(self, player_id: str) -> List[Train]:
        """
        Returns a list of all of the open trains for the given player.

        Args:
            player_id (str): The unique id of the player.

        Returns:
            List[Train]: A list of all of the open trains for the given player.

        Examples:
            >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 4), (4, 2)], player_id="fghij")], engine=(0, 0))
            >>> board.get_open_trains("abcde")
            [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}]
            >>> board.get_open_trains("fghij")
            [{'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 4), (4, 2)], 'is_open': False, 'player_id': 'fghij'}]
        """
        return [train for train in self.trains if train.is_open_for_player(player_id)]

    def open_train(self, player: Player) -> None:
        """
        Opens the given player's train.

        Args:
            player (Player): The player whose train to open.

        Examples:
            >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 4), (4, 2)], player_id="fghij")], engine=(0, 0))
            >>> board.open_train(Player(dominoes=[], player_id="abcde"))
            >>> board.trains
            [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': True, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 4), (4, 2)], 'is_open': False, 'player_id': 'fghij'}]
        """
        for train in self.trains:
            if train.player_id == player.id:
                train.is_open = True

    def close_train(self, player: Player) -> None:
        """
        Closes the given player's train.

        Args:
            player (Player): The player whose train to close.

        Examples:
            >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 4), (4, 2)], player_id="fghij")], engine=(0, 0))
            >>> board.open_train(Player(dominoes=[], player_id="abcde"))
            >>> board.trains
            [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': True, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 4), (4, 2)], 'is_open': False, 'player_id': 'fghij'}]
            >>> board.close_train(Player(dominoes=[], player_id="abcde"))
            >>> board.trains
            [{'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}, {'id': 'kzjxwz', 'dominoes': [(0, 0), (0, 4), (4, 2)], 'is_open': False, 'player_id': 'fghij'}]
        """
        for train in self.trains:
            if train.player_id == player.id:
                train.is_open = False

    def get_train_with_double(self) -> Optional[Train]:
        """
        Returns the train that ends in a double if one exists, otherwise
        returns None.

        Returns:
            Optional[Train]: The train that ends in a double if one exists, otherwise None.

        Examples:
            >>> board = Board(trains=[Train(dominoes=[(0, 0), (0, 1), (1, 1)], player_id="abcde"), Train(dominoes=[(0, 0), (0, 4), (4, 2)], player_id="fghij")], engine=(0, 0))
            >>> board.get_train_with_double()
            {'id': 'xrioqj', 'dominoes': [(0, 0), (0, 1), (1, 1)], 'is_open': False, 'player_id': 'abcde'}
        """
        for train in self.trains:
            if train.ends_in_double():
                return train
        return None

    def make_starting_choices_for_player(
        self, continuations: List[Continuation], player: Player
    ) -> List[Tuple[Domino, Optional[str], Optional[bool]]]:
        """
        Returns a list of all possible starting dominos and corresponding
        train ids for the player's current turn based on a list of
        continuations (playable numbers, and the corresponding open train
        that ends in said number) received as input.

        The player's actual move may be a list of dominoes if they are
        playing a double or if it is their first turn, but this method
        only returns the first domino of each possible move. Any valid
        move the player makes MUST either be in this list or start with a
        domino/train combination in this list.

        Args:
            continuations (List[Continuation]): A list of continuations (playable numbers, and the corresponding open train that ends in said number) for the player's current turn.
            player (Player): The player whose turn it is.

        Returns:
            List[Tuple[Domino, Optional[str], Optional[bool]]]: Each tuple in the list contains:
                - **Domino** (*Domino*): The first domino in a potential move.
                - **Train ID** (*Optional[str]*): The ID of the train on which the domino can be played or `None` if not applicable.
                - **Starts Communal Train** (*Optional[bool]*): True if the move starts a communal train, False for a personal train, and None if not applicable.
        """
        all_starter_choices: List[Tuple[Domino,
                                        Optional[str], Optional[bool]]] = []
        for continuation in continuations:
            end_val = continuation["end_val"]
            train_id = continuation["train_id"]
            starting_mexican_train = continuation["starting_mexican_train"]
            for domino in player.dominoes:
                if domino[0] == end_val or domino[1] == end_val:
                    # A possible starting move only calculates which dominoes can
                    # "start" a train, so a user could actually have a longer list
                    # than the one generated below. That being said, every list
                    # of possible dominoes must "start" with the following domino
                    if domino[0] == end_val:
                        new_domino = domino
                    else:
                        new_domino = (domino[1], domino[0])
                    if train_id is None:
                        possible_choice: Tuple[
                            Domino, Optional[str], Optional[bool]
                        ] = (new_domino, train_id, starting_mexican_train)
                    else:
                        possible_choice = (new_domino, train_id, None)
                    all_starter_choices.append(possible_choice)
        return all_starter_choices

    def get_continuations(self, player: Player) -> List[Continuation]:
        """
        Returns a list of all possible continuations (playable numbers,
        and the corresponding open train that ends in said number) for
        the player's current turn.

        Args:
            player (Player): The player whose turn it is.

        Returns:
            List[Continuation]: A list of continuations (playable numbers, and the corresponding open train that ends in said number) for the player's current turn.
        """
        double_train = self.get_train_with_double()
        # if there is a double on the board, then the only option is for the
        # player to fulfill it
        if double_train is not None:
            return [
                {
                    "end_val": double_train.get_end_value(),
                    "train_id": double_train.id,
                    "starting_mexican_train": False,
                }
            ]

        choices: List[Continuation] = []
        player_has_train = False
        board_has_communal_train = False
        for train in self.trains:
            if train.player_id == player.id:
                player_has_train = True
                # add the continuation at the end of the player's train
                choices.append(
                    {
                        "end_val": train.get_end_value(),
                        "train_id": train.id,
                        "starting_mexican_train": False,
                    }
                )
            elif train.player_id is None:
                board_has_communal_train = True
                # add the continuation at the end of the communal train
                choices.append(
                    {
                        "end_val": train.get_end_value(),
                        "train_id": train.id,
                        "starting_mexican_train": False,
                    }
                )
            elif train.is_open:
                # add the continuation at the end of the open train that
                # belongs to another player
                choices.append(
                    {
                        "end_val": train.get_end_value(),
                        "train_id": train.id,
                        "starting_mexican_train": False,
                    }
                )

        # Record whether the player can make a new train
        if not player_has_train:
            if self.engine is None:
                raise Exception("No engine in the train")
            # add the continuation off of the engine in which the player
            # starts their own train
            choices.append(
                {
                    "end_val": self.engine[1],
                    "train_id": None,
                    "starting_mexican_train": False,
                }
            )

        if player_has_train and (not board_has_communal_train):
            if self.engine is None:
                raise Exception("No engine in the train")
            choices.append(
                {
                    "end_val": self.engine[1],
                    "train_id": None,
                    "starting_mexican_train": True,
                }
            )

        # now get the distinct list of choices
        all_choices = [
            (c["end_val"], c["train_id"], c["starting_mexican_train"]) for c in choices
        ]
        distinct_choices = list(set(all_choices))
        list_of_continuations: List[Continuation] = [
            {"end_val": c[0], "train_id": c[1], "starting_mexican_train": c[2]}
            for c in distinct_choices
        ]

        return list_of_continuations

    def get_choices(
        self, player: Player
    ) -> List[Tuple[Domino, Optional[str], Optional[bool]]]:
        """
        Returns a list of all possible starting dominos, corresponding
        train ids, and if the train id is None then tells whether the new train
        would be a personal or communal train based on options for the player's
        current turn.

        The player's actual move may be a list of dominoes if they are
        playing a double or if it is their first turn, but this method only
        returns the first domino of each possible move. Any valid move the
        player makes MUST either be in this list or start with a
        domino/train combination in this list.

        NOTE: This method may return moves that start with a double that the
        player is capable of fulfilling. Note that this method only returns the
        first domino of each possible move, and if the player tries to play a
        double that they can fulfill (without fulfilling it), an exception will
        be raised in the `handle_player_turn` method of the `MexicanTrain` class.

        Args:
            player (Player): The player whose turn it is.

        Returns:
            List[Tuple[Domino, Optional[str], Optional[bool]]]: Each tuple in the list contains:
                - **Domino** (*Domino*): The first domino in a potential move.
                - **Train ID** (*Optional[str]*): The ID of the train on which the domino can be played or `None` if not applicable.
                - **Starts Communal Train** (*Optional[bool]*): True if the move starts a communal train, False for a personal train, and None if not applicable.
        """
        # print("player", player)
        continuations = self.get_continuations(player)
        # print("continuations", continuations)
        choices = self.make_starting_choices_for_player(continuations, player)
        # print("choices", choices)
        return choices

    def __str__(self):
        """
        A string representation of the board that can be printed to the console

        Returns:
            str: A string representation of the board.
        """
        return str({"trains": self.trains, "engine": self.engine})

    def __repr__(self):
        """
        A representation of the board that can be used for debugging

        Returns:
            str: A representation of the board.
        """
        return str({"trains": self.trains, "engine": self.engine})

