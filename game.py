import random
import string
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict, TypedDict, Callable
import time

start_time = time.time()

Domino = Tuple[int, int]
"""A domino represented as a tuple of two integers indicating the values on each half."""


class GameLogEntry(TypedDict):
    """
    Typed dictionary representing an entry in the game log.
    """

    row_id: int  # integer unique to each row
    turn_id: int  # integer unique to each turn
    player_id: str  # string unique to each player equal to the player's id
    domino_played: Domino | None  # tuple of two integers indicating the values on each half of the domino
    train_played_on: str | None  # string unique to each train equal to the train's id
    is_new_personal_train: bool | None  # boolean indicating whether the train played on was a new personal train
    is_new_mexican_train: bool | None  # boolean indicating whether the train played on was a new mexican train
    picked_up_domino: bool  # boolean indicating whether the player picked up a domino from the boneyard
    did_set_engine: bool  # boolean indicating whether the player set the engine


class Continuation(TypedDict):
    """
    Typed dictionary representing the information needed to continue a train in the game.
    """

    end_val: int  # The value at the end of the train where a new domino can be placed.
    train_id: Optional[
        str
    ]  # The unique identifier for the train, if applicable. If None, then the player is starting a new train off of the 'engine' at the center of the board.
    starting_mexican_train: Optional[
        bool
    ]  # Indicates whether this starts a new Mexican train.


def is_double(domino: Domino) -> bool:
    """
    Checks whether a domino is a double (a 'double' is a domino with the same value on both halves).

    This function takes a domino and returns True if its two halves are equal or False if not.

    Args:
        domino (Domino): The domino to check.

    Returns:
        bool: True if the domino is a double, False otherwise.

    Examples:
        >>> is_double((0, 0))
        True
        >>> is_double((0, 1))
        False
    """
    return domino[0] == domino[1]


# We create a class for Move to handle more nuanced cases
# like when a player has their first turn and there is an
# unfulfilled double on the board. In this case, a valid
# move might actually play dominoes on two different trains.
class DominoesToPlay(TypedDict):
    """
    Typed dictionary representing the information needed to play a sequence of dominoes on a train.
    """

    dominoes: List[Domino]  # The dominoes to play on the train.
    train_id: Optional[
        str
    ]  # The unique identifier for the train. If None, then the player is starting a new train off of the 'engine' at the center of the board.
    starting_mexican_train: Optional[
        bool
    ]  # Indicates whether this starts a new Mexican train.


class Move:
    """
    Representation of a move in the game.

    This class represents a move in the game. A move is a sequence of dominoes
    played on one or more trains. A move can be a player's first turn, in which
    case they can play as many valid dominoes as they want on as many open
    trains as they want (or start their own new train off of the 'engine' in the
    center of the board). A move can also be made on a player's turn after
    their first turn. In this case they can only play one domino on any open
    train, or two dominoes if the first domino is a double and the second
    domino fulfills the double. Finally, a move can be a player passing, in
    which case the move is represented by None.

    Attributes:
        sequences_to_play (Optional[List[DominoesToPlay]]): A list of the sequences of dominoes to play on trains. If None, then the player is passing.

    Raises:
        ValueError: If the move has no sequences to play but is not None.
        ValueError: If the move has an empty list of sequences to play.
        ValueError: If the move has more than one sequence to play and any sequence other than the last one ends in a double.

    Examples:
        >>> move = Move(sequences_to_play=[{"dominoes": [(0, 0)], "train_id": None, "starting_mexican_train": False}])
        >>> move.ends_in_double
        True
        >>> move.all_dominoes_played
        [(0, 0)]
        >>> move.train_ids_played_on
        [None]
        >>> move.starts_with_double
        True
        >>> move.collapse_sequences_by_train_id()
        [([(0, 0)], None)]
        >>> str(move)
        'Player played [(0, 0)] on train None'
        >>> repr(move)
        [{'dominoes': [(0, 0)], 'train_id': None, 'starting_mexican_train': False}]
    """

    def __init__(self, sequences_to_play: Optional[List[DominoesToPlay]] = None):
        """
        Initializes a move.

        Args:
            sequences_to_play (Optional[List[DominoesToPlay]]): A list of the sequences of dominoes to play on trains. If None, then the player is passing.

        Raises:
            ValueError: If the move has no sequences to play but is not None.
            ValueError: If the move has an empty list of sequences to play.
            ValueError: If the move has more than one sequence to play and any sequence other than the last one ends in a double.
        """
        # if sequences_to_play is None, then the player is passing
        if (sequences_to_play is not None) and (len(sequences_to_play) == 0):
            raise ValueError(
                "Move must have at least one domino. If you intend to pass, use None."
            )

        if sequences_to_play is not None:
            # make sure the player doesn't try to play an unfulfilled doule
            # and then keep playing on another train (which conceptually would
            # only make sense if it was the player's first turn and another
            # player who had already gone had an open train)
            for idx, domino_sequence in enumerate(sequences_to_play):
                if is_double(domino_sequence["dominoes"][-1]):
                    if idx < len(sequences_to_play) - 1:
                        raise ValueError(
                            "Cannot play on a subsequent train after playing an unfulfilled double"
                        )

        self.sequences_to_play = sequences_to_play

    @property
    def ends_in_double(self) -> bool:
        """
        Tells whether the move will result in an unfufilled double on the board.

        Returns:
            bool: True if the move ends in a double, False otherwise.
        """
        if self.sequences_to_play is None:
            return False
        return is_double(self.sequences_to_play[-1]["dominoes"][-1])

    @property
    def all_dominoes_played(self) -> List[Domino]:
        """
        Tells us all of the dominoes played in the move regardless of which
        train they were played on.

        Returns:
            List[Domino]: A list of all of the dominoes played in the move.
        """
        output: List[Domino] = []
        if self.sequences_to_play is None:
            return output
        for domino_sequence in self.sequences_to_play:
            output.extend(domino_sequence["dominoes"])
        return output

    @property
    def train_ids_played_on(self) -> List[str]:
        """
        Tells us all of the pre-existing train ids played on in the move.
        Does not include `None` if the player is starting a new train.

        Returns:
            List[str]: A list of all of the train ids played on in the move.
        """
        output: List[str] = []
        if self.sequences_to_play is None:
            return output
        for domino_sequence in self.sequences_to_play:
            if domino_sequence["train_id"] is not None:
                output.append(domino_sequence["train_id"])
        return output

    @property
    def starts_with_double(self) -> bool:
        """
        Tells whether the move starts with a double.

        Returns:
            bool: True if the move starts with a double, False otherwise.
        """
        if self.sequences_to_play is None:
            return False
        return is_double(self.sequences_to_play[0]["dominoes"][0])

    # in the edge case where it is a player's first turn, there is an
    # unfulfilled double, they can fulfill it and they're able to play
    # a string of dominoes that would end in a double, but they also still
    # have a tile left to start a train of their own, then they would
    # have to play that as 3 separate sequences, because they can't start
    # their own train once they've played an unfulfillable double on the
    # other train, so they have to switch from the original train to starting
    # their own and then back to finish with an unfulfilled double on the
    # original train. for this reason, we need an ability to "collapse" the
    # sequences into one sequence per train_id
    def collapse_sequences_by_train_id(
        self,
    ) -> List[Tuple[List[Domino], Optional[str]]]:
        """
        Returns a list of tuples of the form (dominoes, train_id) where the
        dominoes are the dominoes played on the train with the given train_id.

        Returns:
            List[Tuple[List[Domino], Optional[str]]]: Each tuple represents a sequence of dominoes played on a train and contains:
                - **Dominoes** (*List[Domino]*): A list of the dominoes played on the train
                - **Train Id** (*Optional[str]*): The train id of the train the dominoes were played on

        Examples:
            >>> move = Move(sequences_to_play=[{"dominoes": [(0, 0)], "train_id": None, "starting_mexican_train": False}])
            >>> move.collapse_sequences_by_train_id()
            [([(0, 0)], None)]
            >>> move = Move(sequences_to_play=[{"dominoes": [(0, 1)], "train_id": None, "starting_mexican_train": False}, {"dominoes": [(0, 2)], "train_id": "abcde", "starting_mexican_train": False}])
            >>> move.collapse_sequences_by_train_id()
            [([(0, 1)], None), ([(0, 2)], "abcde")]
            >>> move = Move(sequences_to_play=[{"dominoes": [(0, 1)], "train_id": None, "starting_mexican_train": False}, {"dominoes": [(0, 2), (2, 3)], "train_id": "abcde", "starting_mexican_train": False}, {"dominoes": [(1, 1)], "train_id": None, "starting_mexican_train": False}])
            >>> move.collapse_sequences_by_train_id()
            [([(0, 1), (1, 1)], None), ([(0, 2), (2, 3)], "abcde")]
        """
        if self.sequences_to_play is None:
            return []
        sequences_dict: Dict[str, List[Domino]] = {}
        for sequence in self.sequences_to_play:
            if sequence["train_id"] not in sequences_dict:
                sequences_dict[str(sequence["train_id"])] = sequence["dominoes"]
            else:
                sequences_dict[sequence["train_id"]].extend(sequence["dominoes"])
        output: List[Tuple[List[Domino], Optional[str]]] = []
        for train_id, dominoes in sequences_dict.items():
            if train_id == "None":
                output.append((dominoes, None))
            else:
                output.append((dominoes, train_id))
        return output

    def __str__(self):
        """
        A string representation of the move that can be printed to the console
        and tells, in plain English, what the player did.

        Returns:
            str: A string representation of the move.
        """
        if self.sequences_to_play is None:
            return "Player passed"
        if len(self.sequences_to_play) == 1:
            return "Player played {} on train {}".format(
                self.sequences_to_play[0]["dominoes"],
                self.sequences_to_play[0]["train_id"],
            )
        else:
            output = ""
            for idx in range(len(self.sequences_to_play)):
                if idx == 0:
                    output += "Player played {} on train {}".format(
                        self.sequences_to_play[idx]["dominoes"],
                        self.sequences_to_play[idx]["train_id"],
                    )
                else:
                    output += " and {} on train {}".format(
                        self.sequences_to_play[idx]["dominoes"],
                        self.sequences_to_play[idx]["train_id"],
                    )
            return output

    def __repr__(self):
        """
        A string representation of the move that can be printed to the console
        and represents the move as a list of dictionaries.

        Returns:
            str: A string representation of the move.
        """
        return str(self.sequences_to_play)


def canonical(domino: Domino) -> Domino:
    """
    Returns the domino in canonical form, i.e. (0, 1) instead of (1, 0)
    so that every domino has a unique representation.

    Args:
        domino (Domino): The domino to convert to canonical form.

    Returns:
        Domino: The domino in canonical form.

    Examples:
        >>> canonical((0, 1))
        (0, 1)
        >>> canonical((1, 0))
        (0, 1)
    """
    return (min(domino), max(domino))


def make_all_dominoes(max_num_dots: int = 12) -> List[Domino]:
    """
    Returns a list of all of the dominos in a set. The max_num_dots
    parameter determines the maximum number of dots on a domino within
    the set, and therefore the size of the set.

    Args:
        max_num_dots (int): The maximum number of dots on a domino.

    Returns:
        List[Domino]: A list of all of the dominos in the set.
    """
    dominoes: List[Domino] = []
    for i in range(0, max_num_dots + 1):
        for j in range(i, max_num_dots + 1):
            dominoes.append((i, j))
    return dominoes


def shift_arr(arr: List[Any], n: int) -> List[Any]:
    """
    Shifts the first n elements of an array to the end (not in place).

    Args:
        arr (List[Any]): The array to shift.
        n (int): The number of elements to shift.

    Returns:
        List[Any]: The shifted array.

    Examples:
        >>> shift_arr([1, 2, 3, 4, 5], 2)
        [3, 4, 5, 1, 2]
        >>> shift_arr([1, 2, 3, 4, 5], 0)
        [1, 2, 3, 4, 5]
        >>> shift_arr([1, 2, 3, 4, 5], 5)
        [1, 2, 3, 4, 5]
        >>> shift_arr([1, 2, 3, 4, 5], 3)
        [4, 5, 1, 2, 3]
    """
    return arr[n:] + arr[:n]


def random_string(length: int = 12) -> str:
    """
    Creates a random string of lowercase letters of the given length.

    Args:
        length (int): The length of the string to create.

    Returns:
        str: The random string.

    Examples:
        >>> random_string(6)
        'xvzvko'
        >>> random_string(6)
        'jxqzss'
        >>> random_string(4)
        'rmqw'
        >>> random_string()
        'vvqsvapteolk'
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


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

    @property
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
        all_starter_choices: List[Tuple[Domino, Optional[str], Optional[bool]]] = []
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


class MexicanTrain:
    """
    Representation of the Mexican Train game.

    This class represents the Mexican Train game. The game has a list of
    players, a board, and a turn counter.

    Attributes:
        players (List[Player]): The players in the game.
        board (Board): The board in the game.
        is_first (bool): Whether it is the first turn of the game.
        turn (int): The index of the player whose turn it is.
        turn_count (int): The number of turns that have been taken.
        player_count (int): The number of players in the game.
        player_agents (List[MexicanTrainBot]): The agents for each player.
        random_seed (Optional[int]): The random seed for the game.
        dominoes (List[Domino]): The dominoes in the game.
        game_log (List[GameLogEntry]): The log of the game.
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """
        Initializes a Mexican Train game.

        Args:
            random_seed (Optional[int]): The random seed for the game.
        """
        self.players: List[Player] = []
        self.board: Board = Board(trains=[], engine=None)
        self.is_first: bool = False

        self.turn: int = 0
        self.turn_count: int = 0
        self.player_count: int = 0
        self.player_agents: List[MexicanTrainBot] = []
        self.random_seed: Optional[int] = random_seed
        self.dominoes: List[Domino] = make_all_dominoes()
        self.game_log: List[GameLogEntry] = []

    def add_player(self, agent_class: "MexicanTrainBot") -> None:
        """
        Adds a player to the game.

        Args:
            agent_class (MexicanTrainBot): The agent class for the player - this is the bot that plays the game.
        """
        self.player_count += 1
        self.player_agents.append(agent_class)
        self.players.append(Player(dominoes=[], player_id=agent_class.name))

    # Up to 4 players take 15 dominoes each, 5 or 6 take 12 each, 7 or 8 take 10 each.
    def get_hand_size(self) -> int:
        """
        Returns the number of dominoes each player should have in their hand.

        Returns:
            int: The number of dominoes each player should have in their hand.
        """
        if self.player_count < 5:
            return 15
        elif self.player_count < 7:
            return 12
        else:
            return 10

    def deal(self) -> None:
        """
        Deals the dominoes to the players.
        """
        random.seed(self.random_seed)
        random.shuffle(self.dominoes)

        hand_size = self.get_hand_size()
        for i in range(self.player_count):
            hand = self.dominoes[:hand_size]
            self.dominoes = self.dominoes[hand_size:]
            self.players[i].dominoes = hand

    def pickup(self, player: Player) -> Optional[Domino]:
        """
        Picks up a domino from the boneyard.

        Args:
            player (Player): The player who is picking up a domino.

        Returns:
            Optional[Domino]: The domino that the player picked up, if there were any left in the boneyard.
        """
        if len(self.dominoes) == 0:
            # print("No more dominoes to pickup")
            return
        domino = self.dominoes.pop()
        player.dominoes.append(domino)
        return domino

    def valid_domino_sequence(self, dominoes: List[Domino]) -> bool:
        """
        Returns True if the given list of dominoes is a valid sequence
        where the first half of each domino matches the second half of
        the prior domino. Returns False otherwise.

        Args:
            dominoes (List[Domino]): The list of dominoes to check.

        Returns:
            bool: True if the given list of dominoes is a valid sequence, False otherwise.
        """
        if len(dominoes) == 0:
            return False
        if len(dominoes) == 1:
            return True
        for i in range(len(dominoes) - 1):
            if dominoes[i][1] != dominoes[i + 1][0]:
                return False
        return True

    def check_if_player_can_fulfill_double(self, player: Player) -> bool:
        """
        Checks if the player has a domino that can fulfill the double
        currently on the board.

        Args:
            player (Player): The player to check.

        Returns:
            bool: True if the player has a domino that can fulfill the double, False otherwise.
        """
        if not self.board.contains_unfulfilled_double:
            raise Exception("No unfulfilled double")
        if self.board.unfulfilled_double_value is None:
            raise Exception("No unfulfilled double value")
        for domino in player.dominoes:
            if domino[0] == self.board.unfulfilled_double_value:
                return True
            if domino[1] == self.board.unfulfilled_double_value:
                return True
        return False

    def check_valid_move(
        self, player: Player, proposed_move: Optional[Move], is_first: bool = False
    ) -> bool:
        """
        Checks if the proposed move is valid.

        If the proposed move is None, checks if the player is allowed
        to pass. If the proposed move is not None, checks if the
        player is allowed to make the proposed move.

        Args:
            player (Player): The player who is making the move.
            proposed_move (Optional[Move]): The proposed move.
            is_first (bool): Whether it is the first turn of the game.

        Returns:
            bool: True if the proposed move is valid, False otherwise.
        """
        # let the player pass if they're not obligated to fulfill a double
        if proposed_move is None or proposed_move.sequences_to_play is None:
            if self.board.contains_unfulfilled_double:
                if self.check_if_player_can_fulfill_double(player):
                    # they can't pass if they have a domino that can fulfill
                    # the double
                    return False
            return True

        # don't let the player end their first turn in a double, whether or
        # not they can fulfill it (this is for the variant of the game
        # we are playing)
        if is_first and proposed_move.ends_in_double:
            return False

        # don't let the player end their turn in a double they are able to
        # fulfill
        if proposed_move.ends_in_double:
            double_value = proposed_move.sequences_to_play[-1]["dominoes"][-1][1]
            # we use the set of canonical dominoes to avoid not
            # recognizing that, for example, (0, 1) is the same as (1, 0)
            dominoes_played = set(
                [canonical(d) for d in proposed_move.all_dominoes_played]
            )
            dominoes_before_move = set([canonical(d) for d in player.dominoes])
            player_remaining_dominoes = list(dominoes_before_move - dominoes_played)
            player_can_fulfill_double = False
            for domino in player_remaining_dominoes:
                if domino[0] == double_value:
                    player_can_fulfill_double = True
                if domino[1] == double_value:
                    player_can_fulfill_double = True
            if player_can_fulfill_double:
                # they can't end their move on a double if they have a domino
                # that can fulfill it
                return False

        # if the player wants to pass, then they should use None, not an empty
        # list
        if len(proposed_move.sequences_to_play) == 0:
            return False

        # if it's not the player's first turn, then they should only play 1
        # domino on 1 train or 2 dominoes on 1 train if their first domino
        # is a double that they can fulfill
        if (
            (len(proposed_move.all_dominoes_played) > 2)
            or (len(proposed_move.sequences_to_play) > 1)
        ) and (not is_first):
            return False

        # There should rarely be more than one sequence of dominoes in a move,
        # but it's possible on the first turn if another player who has
        # already gone has an open train. We go through each sequence
        # (usually just 1) and check if it's valid. If any sequences are
        # invalid, then the whole move is invalid.
        for proposed_dominoes, _ in proposed_move.collapse_sequences_by_train_id():
            if not self.valid_domino_sequence(proposed_dominoes):
                return False

            # it's not a valid move to "play" a sequence with 0 dominoes
            if len(proposed_dominoes) == 0:
                return False

        # Make sure that if there are 2 dominoes played on a turn after the
        # first turn, then the first domino is a double and the second
        # domino fulfills the double
        if len(proposed_move.all_dominoes_played) == 2 and (not is_first):
            if not proposed_move.starts_with_double:
                return False
            # to have made it here in the code, we know that the proposed move
            # only contains one sequence of dominoes on one train, so we can
            # use the .all_dominoes_played property to get the dominoes
            double_value = proposed_move.all_dominoes_played[0][1]
            fulfills_double = proposed_move.all_dominoes_played[1][0] == double_value
            if not fulfills_double:
                return False

        # Make sure that the first domino we play on any given train is actually
        # a valid choice for that train (i.e. it matches the end value of the
        # train)
        allowed_choices = self.board.get_choices(player)
        for (
            proposed_dominoes,
            proposed_train_id,
        ) in proposed_move.collapse_sequences_by_train_id():
            is_good: bool = False
            first_proposed_domino = proposed_dominoes[0]
            number_of_communal_trains_on_board = len(
                [1 for train in self.board.trains if train.player_id is None]
            )
            for (
                allowed_first_domino,
                open_train_id,
                is_creating_mexican_train,
            ) in allowed_choices:
                if number_of_communal_trains_on_board > 0 and is_creating_mexican_train:
                    # for now we are playing the variant of the game in which
                    # there can be only one (HIGHLANDER reference!!) mexican
                    # train
                    return False
                if proposed_train_id == open_train_id and canonical(
                    first_proposed_domino
                ) == canonical(allowed_first_domino):
                    is_good = True
            if not is_good:
                return False

        # if we've made it this far, then the move is valid
        return True

    def add_to_trains(self, player: Player, move: Move) -> str | None:
        """
        Adds the dominoes in the move to the trains specified by the move.
        If the move creates a new train, then the id of the new train is
        returned.

        Updates the board's `contains_unfulfilled_double`,
        `unfulfilled_double_value`, and `unfulfilled_double_train_id` fields
        if appropriate.

        Args:
            player (Player): The player who is making the move.
            move (Move): The move to make.

        Returns:
            str | None: The id of the new train if the move creates a new train, None otherwise.
        """

        if move.sequences_to_play is None:
            raise Exception("Can't add to a train if no dominoes are played")

        for sequence_to_play in move.sequences_to_play:
            append_dominoes = sequence_to_play["dominoes"]
            train_id = sequence_to_play["train_id"]

            if train_id is None:
                # You wish to make a new Train
                # Will this new train be the Mexican Train? or a personal one?
                if sequence_to_play["starting_mexican_train"]:
                    proposed_train_id = f"mexican_train_started_by_{player.id}"
                    count_of_trains_with_this_id = 0
                    for train in self.board.trains:
                        if train.id.startswith(proposed_train_id):
                            count_of_trains_with_this_id += 1
                    if count_of_trains_with_this_id > 0:
                        proposed_train_id += f"_{count_of_trains_with_this_id + 1}"
                    communal_mexican_train = Train(
                        dominoes=append_dominoes,
                        player_id=None,
                        train_id=proposed_train_id,
                    )
                    self.board.trains.append(communal_mexican_train)
                    player.remove_dominoes(append_dominoes)
                    return proposed_train_id
                else:
                    proposed_train_id = f"{player.id}_personal_train"
                    if proposed_train_id in [train.id for train in self.board.trains]:
                        raise Exception("Player already has a personal train")
                    for train in self.board.trains:
                        if train.player_id == player.id:
                            raise Exception(
                                f"Player already has a personal train called '{train.id}'."
                            )
                    personal_train = Train(
                        dominoes=append_dominoes,
                        player_id=player.id,
                        train_id=proposed_train_id,
                    )
                    self.board.trains.append(personal_train)
                    player.remove_dominoes(append_dominoes)
                    return proposed_train_id
            else:
                for train in self.board.trains:
                    if train.id == train_id:
                        train.add_dominoes(append_dominoes)
                        player.remove_dominoes(append_dominoes)
                        train_ends_in_double_after_move = is_double(append_dominoes[-1])
                        if (
                            train.player_id == player.id
                            and not train_ends_in_double_after_move
                        ):
                            self.board.close_train(player)
                        return

    def set_engine(self) -> bool:
        """
        Sets the first domino of the whole game (the train engine)
        equal to highest double of the first player who has at least
        one double. If no player has a double, returns False.

        Returns:
            bool: True if the engine was found, False otherwise.
        """
        # Check for highest doubles
        for i in range(len(self.players)):
            player = self.players[i]
            highest_double = player.get_highest_double()
            if highest_double is not None:
                self.board.engine = highest_double
                # update the game log
                new_game_log_entry: GameLogEntry = {
                    "row_id": len(self.game_log) + 1,
                    "turn_id": self.turn_count,
                    "player_id": player.id,
                    "domino_played": highest_double,
                    "train_played_on": None,
                    "is_new_personal_train": None,
                    "is_new_mexican_train": None,
                    "picked_up_domino": False,
                    "did_set_engine": True,
                }
                self.game_log.append(new_game_log_entry)
                player.remove_domino(highest_double)
                self.player_agents = shift_arr(self.player_agents, i)
                self.players = shift_arr(self.players, i)
                return True

        return False

    def win_condition(self) -> str | None:
        """
        Checks if any player has won the game.

        Returns:
            str | None: The id of the winning player if there is one, None otherwise.
        """
        for player in self.players:
            if len(player.dominoes) == 0:
                return player.id
        return None

    def perform_move(self, player: Player, move: Move | None, is_first: bool) -> bool:
        """
        Performs the given move for the given player. Returns True if the
        player's turn is over, and False if the player gets to play again.

        Args:
            player (Player): The player who is making the move.
            move (Move | None): The move to make.
            is_first (bool): Whether it is the first turn of the game.

        Returns:
            bool: True if the player's turn is over, and False if the player gets to play again.
        """
        if move is None or move.sequences_to_play is None:
            self.pickup(player)
            # update the game log
            new_game_log_entry: GameLogEntry = {
                "row_id": len(self.game_log) + 1,
                "turn_id": self.turn_count,
                "player_id": player.id,
                "domino_played": None,
                "train_played_on": None,
                "is_new_personal_train": None,
                "is_new_mexican_train": None,
                "picked_up_domino": True,
                "did_set_engine": False,
            }
            self.game_log.append(new_game_log_entry)
            # open the player's train because they passed
            self.board.open_train(player)
            # return False because the player gets to play again after picking
            # up a domino
            return False
        else:
            possible_new_train_id = self.add_to_trains(player, move)
            # update the game log
            for sequence in move.sequences_to_play:
                for domino in sequence["dominoes"]:
                    new_game_log_entry: GameLogEntry = {
                        "row_id": len(self.game_log) + 1,
                        "turn_id": self.turn_count,
                        "player_id": player.id,
                        "domino_played": domino,
                        "train_played_on": sequence["train_id"]
                        if sequence["train_id"] is not None
                        else possible_new_train_id,
                        "is_new_personal_train": sequence["train_id"] is None
                        and not sequence["starting_mexican_train"],
                        "is_new_mexican_train": sequence["train_id"] is None
                        and sequence["starting_mexican_train"],
                        "picked_up_domino": False,
                        "did_set_engine": False,
                    }
                    self.game_log.append(new_game_log_entry)
            if move.ends_in_double:
                train_id_with_double = move.sequences_to_play[-1]["train_id"]
                if train_id_with_double is None:
                    if possible_new_train_id is None:
                        raise Exception("No train id provided for the new train")
                    train_id_with_double = possible_new_train_id
                double_value = move.all_dominoes_played[-1][1]
                new_domino = self.pickup(player)
                # update the game log for the picked up domino
                new_game_log_entry: GameLogEntry = {
                    "row_id": len(self.game_log) + 1,
                    "turn_id": self.turn_count,
                    "player_id": player.id,
                    "domino_played": new_domino,
                    "train_played_on": None,
                    "is_new_personal_train": None,
                    "is_new_mexican_train": None,
                    "picked_up_domino": True,
                    "did_set_engine": False,
                }
                self.game_log.append(new_game_log_entry)
                # if there were no more dominoes to pick up then
                # the player's turn is over and their train is open
                if new_domino is None:
                    self.board.open_train(player)
                    return True
                # if the player picked up a domino that can fulfill
                # the double, then they must play it
                can_fulfill_with_new_domino = False
                if new_domino[0] == double_value:
                    can_fulfill_with_new_domino = True
                if new_domino[1] == double_value:
                    # flip the domino
                    new_domino = (new_domino[1], new_domino[0])
                    can_fulfill_with_new_domino = True
                if can_fulfill_with_new_domino:
                    required_move = Move(
                        sequences_to_play=[
                            {
                                "dominoes": [new_domino],
                                "train_id": train_id_with_double,
                                "starting_mexican_train": False,
                            }
                        ],
                    )
                    self.add_to_trains(player, required_move)
                    # update the game log
                    new_game_log_entry = {
                        "row_id": len(self.game_log) + 1,
                        "turn_id": self.turn_count,
                        "player_id": player.id,
                        "domino_played": new_domino,
                        "train_played_on": train_id_with_double,
                        "is_new_personal_train": None,
                        "is_new_mexican_train": None,
                        "picked_up_domino": False,
                        "did_set_engine": False,
                    }
                    if is_first:
                        # if it's the player's first turn, then they can
                        # continue playing because they fulfilled the double
                        # and may have more valid moves to make
                        return False
                    return True
                # if the player picked up a domino that cannot fulfill
                # the double, then their turn is over and their train
                # is open
                self.board.open_train(player)
            return True

    def handle_player_turn(
        self,
        player: Player,
        agent: "MexicanTrainBot",
        piece_counts: List[Tuple[str, int]],
    ) -> None:
        """
        Handles a player's turn.

        Args:
            player (Player): The player whose turn it is.
            agent (MexicanTrainBot): The agent for the player.
            piece_counts (List[Tuple[str, int]]): The number of dominoes each player has left.

        Raises:
            Exception: If the player makes an invalid move.
        """
        # Get agent move
        move = agent.play(
            player, self.board, self.is_first, piece_counts, self.game_log
        )

        # Check if move is valid
        is_valid = self.check_valid_move(player, move, self.is_first)
        if not is_valid:
            log_invalid_move(self.board, player, move)
            raise Exception("Invalid move")

        # Attempt to perform move
        turn_complete = self.perform_move(player, move, self.is_first)
        if turn_complete:
            return

        # If the user drew a domino, they get to play again (NOTE - if they
        # ended their last move in a double that they couldn't fulfill and drew
        # a domino that could fulfill the double, then the self.perform_move
        # method will have already played that double automatically, as
        # required by the rules of the game. But if it is the player's first
        # turn, then they are still able to move again at this point)
        move = agent.play(
            player, self.board, self.is_first, piece_counts, self.game_log
        )
        is_valid = self.check_valid_move(player, move, self.is_first)
        if not is_valid:
            log_invalid_move(self.board, player, move)
            raise Exception("Invalid move")
        if move is not None:
            self.perform_move(player, move, self.is_first)

    def draw_pieces_until_engine_is_set(self) -> bool:
        """
        If the engine isn't set initially, it means no player has
        a double. In this case, each player draws a domino until
        someone has a double, at which point that player sets the
        engine and goes first.

        Returns:
            bool: True if the engine was set, False otherwise.
        """
        while self.board.engine is None:
            for player in self.players:
                domino = self.pickup(player)
                # update the game log
                new_game_log_entry: GameLogEntry = {
                    "row_id": len(self.game_log) + 1,
                    "turn_id": self.turn_count,
                    "player_id": player.id,
                    "domino_played": None,
                    "train_played_on": None,
                    "is_new_personal_train": None,
                    "is_new_mexican_train": None,
                    "picked_up_domino": True,
                    "did_set_engine": False,
                }
                if domino is None:
                    raise Exception("No more dominoes to pickup")
                self.game_log.append(new_game_log_entry)
                if is_double(domino):
                    self.board.engine = domino
                    player.remove_domino(domino)
                    # update the game log
                    new_game_log_entry = {
                        "row_id": len(self.game_log) + 1,
                        "turn_id": self.turn_count,
                        "player_id": player.id,
                        "domino_played": domino,
                        "train_played_on": None,
                        "is_new_personal_train": None,
                        "is_new_mexican_train": None,
                        "picked_up_domino": False,
                        "did_set_engine": True,
                    }
                    self.game_log.append(new_game_log_entry)
                    # don't update the turn count because while the engine
                    # has been set, no player has taken their first turn yet
                    return True
        # we should really never get here - if we do, then something has gone
        # wrong because the engine should have been set by now
        return False

    def play(self):
        """
        Plays the game.

        Returns:
            Player: The winning player.

        Raises:
            Exception: If no engine is found.
        """
        self.deal()
        found_it = self.set_engine()

        if not found_it:
            engine_was_set = self.draw_pieces_until_engine_is_set()
            if not engine_was_set:
                raise Exception("No engine found")

        while True:
            # print(self.board)
            # print(self.players)
            self.is_first = self.turn_count < self.player_count
            player_agent = self.player_agents[self.turn]
            cur_player = self.players[self.turn]
            piece_counts = [(p.id, p.pieces_left) for p in self.players]
            # This is where your agent runs
            self.handle_player_turn(cur_player, player_agent, piece_counts)

            self.turn = (self.turn + 1) % self.player_count
            self.turn_count += 1

            # print(self.board)
            winner = self.win_condition()
            if winner is not None:
                break

            if self.turn_count > 1000:
                # print("Game over, no winner")
                return None

        # print("The winner is player " + str(winner))

        player_results: List[Tuple[str, int]] = []
        for player in self.players:
            if player.id == winner:
                player_score = 0
            else:
                player_score = sum([sum(d) for d in player.dominoes])
            player_results.append((player.id, player_score))
        return sorted(player_results, key=lambda x: x[1])

    def __str__(self):
        """
        A string representation of the game that can be printed to the console

        Returns:
            str: A string representation of the game.
        """
        return str(
            {
                "board": self.board,
                "players": self.players,
                "turn": self.turn,
                "turn_count": self.turn_count,
                "player_count": self.player_count,
                "random_seed": self.random_seed,
                "dominoes": self.dominoes,
            }
        )

    def __repr__(self):
        """
        A representation of the game that can be used for debugging

        Returns:
            str: A representation of the game.
        """
        return str(
            {
                "board": self.board,
                "players": self.players,
                "turn": self.turn,
                "turn_count": self.turn_count,
                "player_count": self.player_count,
                "random_seed": self.random_seed,
                "dominoes": self.dominoes,
            }
        )


def log_invalid_move(board: Board, player: Player, move: Optional[Move]) -> None:
    """
    Logs an invalid move to the console. This is useful for debugging.

    Args:
        board (Board): The board in the game.
        player (Player): The player who made the invalid move.
        move (Optional[Move]): The invalid move.
    """
    print("Invalid move. Details below:")
    print("Board:")
    print(board.__str__())
    print("--------------------")
    print("Player:")
    print(player.__str__())
    print("--------------------")
    print("Move:")
    print(move.__str__())
    print("--------------------")


class EloRating:
    """
    Utility class that allows us to rate the bots based on their performance
    against each other.

    Attributes:
        k (float): The k factor to use for the Elo rating system.
        ratings (Dict[str, float]): The ratings for each bot. The key is the
            name of the bot and the value is the rating.
        iter_count (int): A utility variable used to iterate over the ratings. Should not be used directly.
    """

    def __init__(self, k: float = 32, bot_names: List[str] = []):
        """
        Initializes the EloRating class.

        Args:
            k (float): The k factor to use for the Elo rating system.
        """
        self.k: float = k
        self.ratings: Dict[str, float] = {}
        for bot_name in bot_names:
            self.ratings[bot_name] = 1500.0
        self.iter_count: int = 0

    def get_expected_score(self, bot1: str, bot2: str) -> float:
        """
        Returns the expected score for bot1 against bot2.

        Args:
            bot1 (str): The name of the first bot.
            bot2 (str): The name of the second bot.

        Returns:
            float: The expected score for bot1 against bot2.
        """
        rating1 = self.ratings[bot1]
        rating2 = self.ratings[bot2]
        return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))

    def get_new_ratings_two_players(
        self,
        bot1: str,
        bot2: str,
        bot1_won: Optional[bool] = None,
        tie: Optional[bool] = None,
    ) -> Tuple[float, float]:
        """
        Returns the new ratings for bot1 and bot2 after a game between the two
        bots.

        Args:
            bot1 (str): The name of the first bot.
            bot2 (str): The name of the second bot.
            bot1_won (Optional[bool]): Whether bot1 won the game.
            tie (Optional[bool]): Whether the game was a tie.

        Returns:
            Tuple[float, float]: The new ratings for bot1 and bot2, respectively.

        Raises:
            Exception: If neither bot1_won nor tie is None.
            Exception: If both bot1_won and tie are None.
        """
        if bot1_won is None and tie is None:
            raise Exception("Either bot1_won or tie must be non-None")
        if bot1_won is not None and tie is not None:
            raise Exception("Either bot1_won or tie must be None")
        expected_score = self.get_expected_score(bot1, bot2)
        if bot1_won:
            actual_score = 1.0
        elif bot1_won is None and tie:
            actual_score = 0.5
        else:
            actual_score = 0.0
        new_bot1_rating = self.ratings[bot1] + self.k * (actual_score - expected_score)
        new_bot2_rating = self.ratings[bot2] + self.k * (expected_score - actual_score)
        return (new_bot1_rating, new_bot2_rating)

    def update_ratings(self, bot_names: List[str], bot_scores: List[float]) -> None:
        """
        Updates the ratings for all bots based on the results of a game
        between all of the bots.

        Args:
            bot_names (List[str]): The names of all of the bots.
            bot_scores (List[float]): The scores of all of the bots in the game.
        """
        new_ratings: Dict[str, List[float]] = {}
        for bot_name in bot_names:
            new_ratings[bot_name] = []
        for i in range(len(bot_names)):
            for j in range(i + 1, len(bot_names)):
                bot1 = bot_names[i]
                bot2 = bot_names[j]
                bot1_won = (
                    bot_scores[i] < bot_scores[j]
                )  # lower score is better in Mexican Train
                tie = bot_scores[i] == bot_scores[j]
                if bot1_won:
                    tie = None
                if tie:
                    bot1_won = None
                if bot1_won == False and tie == False:
                    tie = None
                new_bot1_rating, new_bot2_rating = self.get_new_ratings_two_players(
                    bot1, bot2, bot1_won, tie
                )
                new_ratings[bot1].append(new_bot1_rating)
                new_ratings[bot2].append(new_bot2_rating)
        for bot_name in bot_names:
            self.ratings[bot_name] = sum(new_ratings[bot_name]) / len(
                new_ratings[bot_name]
            )
        return

    def __iter__(self):
        """
        Allows us to iterate over the ratings.

        Returns:
            Iterator[Tuple[str, float]]: An iterator over the ratings.
        """
        return self

    def __next__(self):
        """
        Allows us to iterate over the ratings sorted from highest to lowest.

        Returns:
            Tuple[str, float]: The next rating. Each tuple contains:
                - **Name** (*str*) -- The name of the bot.
                - **Rating** (*float*) -- The ELO rating of the bot.
        """
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[0], reverse=True)
        if self.iter_count >= len(self.ratings):
            self.iter_count = 0
            raise StopIteration
        else:
            self.iter_count += 1
            return (
                sorted_ratings[self.iter_count - 1][0],
                sorted_ratings[self.iter_count - 1][1],
            )

    def __len__(self):
        """
        Returns the number of ratings.

        Returns:
            int: The number of ratings.
        """
        return len(self.ratings)


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

    def __init__(self, name: Optional[str]):
        """
        Initializes a Mexican Train bot.

        Args:
            name (Optional[str]): The name of the bot.
        """
        if name is None:
            self.name = random_string()
        else:
            self.name = name

    @abstractmethod
    def play(
        self,
        player: Player,
        board: Board,
        is_first: bool,
        piece_counts: List[Tuple[str, int]],
        game_log: List[GameLogEntry],
    ) -> Optional[Move]:
        """
        The method to choose a `Move` to play. All player agents must implement
        this method.
        """
        pass


# This is a random agent that plays a random move
class RandomPlayerAgent(MexicanTrainBot):
    """
    Representation of a random Mexican Train bot.

    This class represents a random Mexican Train bot. It plays a random
    move from the list of valid moves each turn. On the first turn it
    only plays a single domino at random. it doesn't pick a whole sequence
    of dominoes to play.

    Attributes:
        name (str): The name of the bot.
    """

    def play(
        self,
        player: Player,
        board: Board,
        is_first: bool,
        piece_counts: List[Tuple[str, int]],
        game_log: List[GameLogEntry],
    ) -> Optional[Move]:
        """
        Chooses a random valid move to play when it is the bot's turn.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.
            is_first (bool): Whether it is the first turn of the game.
            piece_counts (List[Tuple[str, int]]): The number of dominoes each player has left.
            game_log (List[GameLogEntry]): The game log.

        Returns:
            Optional[Move]: The move to play.
        """
        choices = board.get_choices(player)

        # if it's the first turn the player can't create a mexican train
        # or end their turn in a double
        if is_first:
            choices = [
                choice
                for choice in choices
                if ((not is_double(choice[0])) and (not choice[2]))
            ]

        if len(choices) == 0:
            return None

        random.shuffle(choices)

        random_choice = choices[0]
        move = Move(
            sequences_to_play=[
                {
                    "dominoes": [random_choice[0]],
                    "train_id": random_choice[1],
                    "starting_mexican_train": random_choice[2],
                }
            ]
        )
        # To avoid an exception in the `handle_player_turn` method of the
        # `MexicanTrain` class, we must ensure that the move doesn't end
        # in a double if the player has a domino that can fulfill it
        if not move.ends_in_double:
            return move
        else:
            double_value = move.all_dominoes_played[-1][1]
            pre_move_dominoes = set([canonical(d) for d in player.dominoes])
            dominoes_used_in_move = set(
                [canonical(d) for d in move.all_dominoes_played]
            )
            player_remaining_dominoes = list(pre_move_dominoes - dominoes_used_in_move)
            random.shuffle(player_remaining_dominoes)
            tile_to_fullfill_double = None
            for domino in player_remaining_dominoes:
                if domino[0] == double_value:
                    tile_to_fullfill_double = domino
                if domino[1] == double_value:
                    flipped_domino = (domino[1], domino[0])
                    tile_to_fullfill_double = flipped_domino
                if tile_to_fullfill_double is not None:
                    if move.sequences_to_play is None:
                        raise Exception(
                            "Move can't end in double if it doesn't play any dominoes"
                        )
                    move.sequences_to_play[-1]["dominoes"].append(
                        tile_to_fullfill_double
                    )
                    return move
            # if we've made it this far, then the player has no dominoes
            # that can fulfill the double, which means it's a valid move
            # to play the double
            return move


class NormalPersonPlayerAgent(MexicanTrainBot):
    """
    Representation of a Mexican Train Bot that plays by a hard-coded set
    of rules that are probably fairly representative of how a normal person
    would play the game.

    Attributes:
        name (str): The name of the bot.
    """

    def construct_first_turn_sequence(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Constructs a valid move for the first turn of the game using a naive
        approach to construct the "longest possible" valid sequence.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Move: The move to play.

        Raises:
            Exception: If the board has no engine.
        """
        if board.engine is None:
            raise Exception("No engine found when constructing first turn sequence")

        starting_value = board.engine[1]

        candidate_dominoes = [
            d if d[0] == starting_value else (d[1], d[0])
            for d in player.dominoes
            if starting_value in d
        ]
        if len(candidate_dominoes) == 0:
            return None

        get_num_second_choices: Callable[[Domino], int] = lambda d: len(
            [
                d2
                for d2 in player.dominoes
                if canonical(d2) != canonical(d) and d[1] in d2
            ]
        )
        num_second_choices = [get_num_second_choices(d) for d in candidate_dominoes]

        # choose the candidate domino with the most second choices
        chosen_domino = candidate_dominoes[
            num_second_choices.index(max(num_second_choices))
        ]

        sequence_of_dominoes = [chosen_domino]
        # keep adding dominoes to the sequence at random until we can't anymore
        pre_turn_dominoes = set([canonical(d) for d in player.dominoes])
        while True:
            end_val = sequence_of_dominoes[-1][1]
            played_dominoes = set([canonical(d) for d in sequence_of_dominoes])
            player_remaining_dominoes = list(pre_turn_dominoes - played_dominoes)
            next_domino_choices = [
                d if d[0] == end_val else (d[1], d[0])
                for d in player_remaining_dominoes
                if end_val in d
            ]
            if len(next_domino_choices) == 0:
                break
            random.shuffle(next_domino_choices)
            next_domino = next_domino_choices[0]
            sequence_of_dominoes.append(next_domino)
        # remove the last domino if it's a double
        if is_double(sequence_of_dominoes[-1]):
            sequence_of_dominoes = sequence_of_dominoes[:-1]
        return Move(
            sequences_to_play=[
                {
                    "dominoes": sequence_of_dominoes,
                    "train_id": None,
                    "starting_mexican_train": False,
                }
            ]
        )

    def check_if_there_is_a_player_who_could_win_scarily_soon(
        self, player: Player, piece_counts: List[Tuple[str, int]]
    ) -> bool:
        """
        Checks if there is a player who has 2 or fewer dominoes while the current
        player has more than 2 OR if there is a player who has 3 dominoes while
        the current player has more than 5.

        Args:
            player (Player): The player whose turn it is.
            piece_counts (List[Tuple[str, int]]): The number of dominoes each player has left.

        Returns:
            bool: True if there is a player who could win the game in the next turn, False otherwise.
        """
        min_dominoes_for_a_player = min([p[1] for p in piece_counts])
        this_player_num_dominoes = len(player.dominoes)
        if (min_dominoes_for_a_player <= 2) and (this_player_num_dominoes > 2):
            return True
        if (min_dominoes_for_a_player <= 3) and (this_player_num_dominoes > 5):
            return True
        return False

    def get_highest_single_point_total_domino_player_can_play(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Checks the possible choices a player has. Looks at the highest point total
        single domino the player can play.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Optional[Move]: The move to play.
        """
        choices = board.get_choices(player)
        non_double_choices = [c for c in choices if not is_double(c[0])]
        if len(non_double_choices) == 0:
            return None
        point_totals = [sum(choice[0]) for choice in non_double_choices]
        highest_point_total_domino = non_double_choices[
            point_totals.index(max(point_totals))
        ]
        return Move(
            sequences_to_play=[
                {
                    "dominoes": [highest_point_total_domino[0]],
                    "train_id": highest_point_total_domino[1],
                    "starting_mexican_train": highest_point_total_domino[2],
                }
            ]
        )

    def get_highest_double_player_can_play(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Checks the possible choices a player has. Looks at the highest point
        total double the player can play.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Optional[Move]: The move to play.
        """
        choices = board.get_choices(player)
        double_choices = [c for c in choices if is_double(c[0])]
        if len(double_choices) == 0:
            return None
        double_values = [choice[0][1] for choice in double_choices]
        point_totals: List[int] = []
        fulfill_dominoes: List[Domino | None] = []
        for value in double_values:
            candidates_to_fulfill_double = [
                d if d[0] == value else (d[1], d[0])
                for d in player.dominoes
                if value in d and not is_double(d)
            ]
            if len(candidates_to_fulfill_double) == 0:
                point_totals.append(2 * value)
                fulfill_dominoes.append(None)
            else:
                fulfill_domino_points = [sum(d) for d in candidates_to_fulfill_double]
                highest_point_domino_to_fulfill_double = candidates_to_fulfill_double[
                    fulfill_domino_points.index(max(fulfill_domino_points))
                ]
                point_totals.append(
                    sum(highest_point_domino_to_fulfill_double) + 2 * value
                )
                fulfill_dominoes.append(highest_point_domino_to_fulfill_double)
        highest_point_total_double = double_choices[
            point_totals.index(max(point_totals))
        ]
        fulfill_domino = fulfill_dominoes[point_totals.index(max(point_totals))]
        return Move(
            sequences_to_play=[
                {
                    "dominoes": [highest_point_total_double[0]]
                    if fulfill_domino is None
                    else [highest_point_total_double[0], fulfill_domino],
                    "train_id": highest_point_total_double[1],
                    "starting_mexican_train": highest_point_total_double[2],
                }
            ]
        )

    def get_move_that_plays_highest_point_total_sequence(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Checks the possible choices a player has and finds the highest point
        total sequence the player can play. This ignores the possibility that
        if the player plays an unfulfilled double, then they might draw from
        the boneyard and still be able to unfulfill the double, which would
        mean the total decrease in points in their hand is less than the
        point total of the sequence they played.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Optional[Move]: The move to play.
        """
        best_single_domino_move = (
            self.get_highest_single_point_total_domino_player_can_play(player, board)
        )
        best_double_move = self.get_highest_double_player_can_play(player, board)
        if best_single_domino_move is None and best_double_move is None:
            return None
        if best_single_domino_move is None:
            return best_double_move
        if best_double_move is None:
            return best_single_domino_move

        assert best_single_domino_move.sequences_to_play is not None
        single_domino_point_total = sum(
            best_single_domino_move.sequences_to_play[0]["dominoes"][0]
        )
        assert best_double_move.sequences_to_play is not None
        double_point_total = sum(
            [sum(d) for d in best_double_move.sequences_to_play[0]["dominoes"]]
        )
        if double_point_total > single_domino_point_total:
            return best_double_move
        return best_single_domino_move

    def find_non_double_that_closes_train(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Checks the possible choices a player has and finds the first non-double
        that closes the player's train.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Optional[Move]: The move to play, or None if no such move exists in which a non-double closes the player's train.
        """
        choices = board.get_choices(player)
        non_double_choices = [c for c in choices if not is_double(c[0])]
        if len(non_double_choices) == 0:
            return None
        for choice in non_double_choices:
            chosen_move_train_id = choice[1]
            for train in board.trains:
                if train.player_id == player.id and train.id == chosen_move_train_id:
                    return Move(
                        sequences_to_play=[
                            {
                                "dominoes": [choice[0]],
                                "train_id": chosen_move_train_id,
                                "starting_mexican_train": choice[2],
                            }
                        ]
                    )
        return None

    def find_double_that_closes_train(
        self, player: Player, board: Board
    ) -> Move | None:
        """
        Checks the possible choices a player has and finds the first double
        that can be played AND fulfilled on the player's personal train.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.

        Returns:
            Optional[Move]: The move to play, or None if no such move exists in which a fulfilled double closes the player's train.
        """
        choices = board.get_choices(player)
        personal_train_id = None
        for train in board.trains:
            if train.player_id == player.id:
                personal_train_id = train.id
                break
        if personal_train_id is None:
            return None
        # find doubles that can be played on the player's personal train
        double_choices = [
            c for c in choices if is_double(c[0]) and c[1] == personal_train_id
        ]
        if len(double_choices) == 0:
            return None

        for choice in double_choices:
            dominoes_that_can_fulfill_double = [
                d if d[0] == choice[0][1] else (d[1], d[0])
                for d in player.dominoes
                if canonical(d) != canonical(choice[0]) and choice[0][1] in d
            ]
            if len(dominoes_that_can_fulfill_double) == 0:
                continue
            domino_to_fulfill_double = dominoes_that_can_fulfill_double[0]
            return Move(
                sequences_to_play=[
                    {
                        "dominoes": [choice[0], domino_to_fulfill_double],
                        "train_id": choice[1],
                        "starting_mexican_train": choice[2],
                    }
                ]
            )
        # if we've made it this far there is a playable double but no domino
        # to fulfill it, which means there's no double to play that will
        # definitively close the player's train
        return None

    def play(
        self,
        player: Player,
        board: Board,
        is_first: bool,
        piece_counts: List[Tuple[str, int]],
        game_log: List[GameLogEntry],
    ) -> Optional[Move]:
        """
        Chooses a valid move to play when it is the bot's turn.

        Args:
            player (Player): The player whose turn it is.
            board (Board): The board in the game.
            is_first (bool): Whether it is the first turn of the game.
            piece_counts (List[Tuple[str, int]]): The number of dominoes each player has left.
            game_log (List[GameLogEntry]): The game log.

        Returns:
            Optional[Move]: The move to play.
        """
        if is_first:
            return self.construct_first_turn_sequence(player, board)

        choices = board.get_choices(player)

        if len(choices) == 0:
            return None

        # if we're afraid that someone could win the game soon, then we should
        # prioritize getting rid of a high point total domino
        if self.check_if_there_is_a_player_who_could_win_scarily_soon(
            player, piece_counts
        ):
            best_move = self.get_move_that_plays_highest_point_total_sequence(
                player, board
            )
            assert (
                best_move is not None
            ), "No valid move found even though we thought there was one because `choices` is not empty"
            return best_move

        # if we made it this far, figure out if the player has their own train
        player_train_is_open = False
        for train in board.trains:
            if train.player_id == player.id:
                player_train_is_open = train.is_open
                break

        # if the player didn't prioritize getting rid of a high point total
        # domino because they thought there would be a lot of time left, then
        # they should close their train if it is open and they have a domino
        # that can close it, or otherwise play a random move.
        if player_train_is_open:
            double_that_closes_train = self.find_double_that_closes_train(player, board)
            non_double_that_closes_train = self.find_non_double_that_closes_train(
                player, board
            )
            if (
                double_that_closes_train is None
                and non_double_that_closes_train is not None
            ):
                return non_double_that_closes_train
            if (
                double_that_closes_train is not None
                and non_double_that_closes_train is None
            ):
                return double_that_closes_train
            if (
                double_that_closes_train is not None
                and non_double_that_closes_train is not None
            ):
                # its probably better for the player to play a double that
                # closes their train, and this is a decent but not perfect
                # player, so with 80% probability they will play the double
                # and with 20% probability they will play the non-double
                if random.random() < 0.8:
                    return double_that_closes_train
                else:
                    return non_double_that_closes_train

        # if we've made it this far then the player's train is open (or
        # they don't even have a personal train) but they don't have a
        # domino that can close it, so they should just play a random
        # move
        random.shuffle(choices)
        random_choice = choices[0]
        sequence_of_dominoes = [random_choice[0]]
        # if random_choice[0] is a fulfillable double, then the player must
        # fulfill it. otherwise it's ok to play a random domino
        if is_double(random_choice[0]):
            double_value = random_choice[0][1]
            dominoes_that_can_fulfill_double = [
                d if d[0] == double_value else (d[1], d[0])
                for d in player.dominoes
                if double_value in d and canonical(d) != canonical(random_choice[0])
            ]
            if len(dominoes_that_can_fulfill_double) > 0:
                random.shuffle(dominoes_that_can_fulfill_double)
                sequence_of_dominoes.append(dominoes_that_can_fulfill_double[0])
        return Move(
            sequences_to_play=[
                {
                    "dominoes": sequence_of_dominoes,
                    "train_id": random_choice[1],
                    "starting_mexican_train": random_choice[2],
                }
            ]
        )


# List of all bots in the tournament
players: List[MexicanTrainBot] = [
    NormalPersonPlayerAgent("John"),
    RandomPlayerAgent("Jacob"),
    RandomPlayerAgent("Jingleheimer"),
    RandomPlayerAgent("Schmidt"),
    NormalPersonPlayerAgent("His"),
    RandomPlayerAgent("Name_1"),
    RandomPlayerAgent("Is"),
    RandomPlayerAgent("My"),
    RandomPlayerAgent("Name_2"),
    RandomPlayerAgent("Too"),
]


def play_1_game(players: List[MexicanTrainBot]) -> list[Tuple[str, int]] | None:
    """
    Plays a single game between the given players.

    Args:
        players (List[MexicanTrainBot]): The players in the game.

    Returns:
        list[Tuple[str, int]] | None: The results of the game if there was a winner, None otherwise. If there was a winner, each tuple in the list contains:
            - **Player Name** (*str*) -- The name of the player.
            - **Player Score** (*int*) -- The score of the player.
    """
    mexican_train = MexicanTrain()
    for player in players:
        mexican_train.add_player(player)
    return mexican_train.play()


def run_simulation_and_produce_ratings(
    player_bots: List[MexicanTrainBot], n_games: int = 1500
):
    """
    Runs a simulation of 4 * `n_games` individual games broken up into `n_games` games of
    4 rounds each. After each round, the ratings are updated based on the
    results of the 4 games. The number of players and actual players in each
    game are chosen randomly. Returns the ELO ratings of each bot at the end of
    the simulation.

    Args:
        player_bots (List[MexicanTrainBot]): The players in the game.
        n_games (int): The number of games to play.

    Returns:
        EloRating (EloRating): The ELO ratings of each bot at the end of the simulation.
    """
    # Create the EloRating object that will rate the players in the tournament
    elo_rating = EloRating(bot_names=[player.name for player in players], k=20)

    for _ in range(n_games):
        # Set up a game with a random number of players
        num_players = random.randint(3, min(len(player_bots), 8))
        randomly_sampled_players = random.sample(player_bots, num_players)
        game_1_result = play_1_game(randomly_sampled_players)
        game_2_result = play_1_game(randomly_sampled_players)
        game_3_result = play_1_game(randomly_sampled_players)
        game_4_result = play_1_game(randomly_sampled_players)
        # aggregate the scores from the 4 games
        aggregate_scores: Dict[str, int] = {
            player.name: 0 for player in randomly_sampled_players
        }
        for game_result in [game_1_result, game_2_result, game_3_result, game_4_result]:
            if game_result is not None:
                for player_name, score in game_result:
                    aggregate_scores[player_name] += score

        # update the ratings based on the results of the 3 games
        elo_rating.update_ratings(
            [player.name for player in randomly_sampled_players],
            [aggregate_scores[player.name] for player in randomly_sampled_players],
        )
    return elo_rating


ratings_by_bot: Dict[str, List[float]] = {}
for player in players:
    ratings_by_bot[player.name] = []


def update_ratings(
    ratings_by_bot: Dict[str, List[float]], elo_rating: EloRating | None
) -> None:
    if elo_rating is None:
        print("No elo rating found")
        return
    for bot_name, rating in elo_rating:
        ratings_by_bot[bot_name].append(rating)


ratings_1 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_1)
ratings_2 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_2)
ratings_3 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_3)
ratings_4 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_4)
ratings_5 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_5)
ratings_6 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_6)
ratings_7 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_7)
ratings_8 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_8)
ratings_9 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_9)
ratings_10 = run_simulation_and_produce_ratings(players)
update_ratings(ratings_by_bot, ratings_10)


def average_elo_ratings_without_outliers(
    ratings_by_bot: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Returns the average ELO rating for each bot after removing the 2 highest and
    2 lowest rating for each bot.

    Args:
        ratings_by_bot (Dict[str,List[float]]): The ELO ratings for each bot.

    Returns:
        Dict[str,float]: The average ELO rating for each bot after removing the highest and lowest rating for each bot.
    """
    average_ratings: Dict[str, float] = {}
    for bot_name, ratings in ratings_by_bot.items():
        ratings_without_outliers = sorted(ratings)[2:-2]
        average_ratings[bot_name] = sum(ratings_without_outliers) / len(
            ratings_without_outliers
        )
    return average_ratings


average_ratings = average_elo_ratings_without_outliers(ratings_by_bot)

# list of tuples of the form (bot_name, average_rating)
sorted_average_ratings = sorted(
    average_ratings.items(), key=lambda x: x[1], reverse=True
)

# print the results in the format: "Place _ - Bot _. ELO Rating: _"
for i in range(len(sorted_average_ratings)):
    print(
        f"Place {i+1} - Bot {sorted_average_ratings[i][0]}. ELO Rating: {sorted_average_ratings[i][1]}"
    )

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")
