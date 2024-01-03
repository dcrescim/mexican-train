from typing import Tuple, TypedDict, Optional, List, Any
import random
import string

Domino = Tuple[int, int]
"""A domino represented as a tuple of two integers indicating the values on each half."""


class Continuation(TypedDict):
    """
    Typed dictionary representing the information needed to continue a train in the game.
    """

    # The value at the end of the train where a new domino can be placed.
    end_val: int
    train_id: Optional[
        str
    ]  # The unique identifier for the train, if applicable. If None, then the player is starting a new train off of the 'engine' at the center of the board.
    starting_mexican_train: Optional[
        bool
    ]  # Indicates whether this starts a new Mexican train.


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


class GameLogEntry(TypedDict):
    """
    Typed dictionary representing an entry in the game log.
    """

    row_id: int  # integer unique to each row
    turn_id: int  # integer unique to each turn
    player_id: str  # string unique to each player equal to the player's id
    # tuple of two integers indicating the values on each half of the domino
    domino_played: Domino | None
    train_played_on: str | None  # string unique to each train equal to the train's id
    # boolean indicating whether the train played on was a new personal train
    is_new_personal_train: bool | None
    # boolean indicating whether the train played on was a new mexican train
    is_new_mexican_train: bool | None
    # boolean indicating whether the player picked up a domino from the boneyard
    picked_up_domino: bool
    did_set_engine: bool  # boolean indicating whether the player set the engine


def canonical(domino: Domino) -> Domino:
    """
    Returns the domino in canonical form, i.e. (0, 1) instead of (1, 0)
    so that every domino has a unique representation. The Canonical form
    is always monotonically increasing (lowest number first)

    Args:
        domino (Domino): The domino to convert to canonical form.

    Returns:
        Domino: The domino in canonical form.

    Examples:
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
        >>> random_string()
        'vvqsvapteolk'
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


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
