import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from typing import Optional, List, Tuple, Dict
from mexican_train.domino_types import DominoesToPlay, Domino, is_double


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
                sequences_dict[str(sequence["train_id"])
                               ] = sequence["dominoes"]
            else:
                sequences_dict[sequence["train_id"]].extend(
                    sequence["dominoes"])
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

