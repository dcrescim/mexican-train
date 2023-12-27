import random
import string
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict, TypedDict

# a domino is a tuple of two integers that represent the values on
# each half of the domino
Domino = Tuple[int, int]


class Continuation(TypedDict):
    end_val: int
    train_id: Optional[str]
    starting_mexican_train: Optional[bool]


def is_double(domino: Domino) -> bool:
    """
    Returns True if the given domino is a double, i.e. if both sides
    have the same number of dots.
    """
    return domino[0] == domino[1]


# We create a class for Move to handle more nuanced cases
# like when a player has their first turn and there is an
# unfulfilled double on the board. In this case, a valid
# move might actually play dominoes on two different trains.
class DominoesToPlay(TypedDict):
    dominoes: List[Domino]
    train_id: Optional[str]
    starting_mexican_train: Optional[bool]


class Move:
    def __init__(self, sequences_to_play: Optional[List[DominoesToPlay]] = None):
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
        if self.sequences_to_play is None:
            return False
        return is_double(self.sequences_to_play[-1]["dominoes"][-1])

    @property
    def all_dominoes_played(self) -> List[Domino]:
        output: List[Domino] = []
        if self.sequences_to_play is None:
            return output
        for domino_sequence in self.sequences_to_play:
            output.extend(domino_sequence["dominoes"])
        return output

    @property
    def train_ids_played_on(self) -> List[str]:
        output: List[str] = []
        if self.sequences_to_play is None:
            return output
        for domino_sequence in self.sequences_to_play:
            if domino_sequence["train_id"] is not None:
                output.append(domino_sequence["train_id"])
        return output

    @property
    def starts_with_double(self) -> bool:
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
        return str(self.sequences_to_play)


def canonical(domino: Domino) -> Domino:
    """
    Returns the domino in canonical form, i.e. (0, 1) instead of (1, 0)
    so that every domino has a unique representation.
    """
    return (min(domino), max(domino))


def make_all_dominoes(max_num_dots: int = 12) -> List[Domino]:
    """
    Returns a list of all of the dominos in a set. The max_num_dots
    parameter determines the maximum number of dots on a domino within
    the set, and therefore the size of the set.
    """
    dominoes: List[Domino] = []
    for i in range(0, max_num_dots + 1):
        for j in range(i, max_num_dots + 1):
            dominoes.append((i, j))
    return dominoes


def shift_arr(arr: List[Any], n: int) -> List[Any]:
    """
    Shifts the first n elements of an array to the end.
    """
    return arr[n:] + arr[:n]


def random_string(length: int = 12) -> str:
    """
    Returns a random string of lowercase letters of the given length.
    """
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


class Player:
    def __init__(self, dominoes: List[Domino] = [], player_id: Optional[str] = None):
        if player_id is None:
            self.id = random_string(6)
        else:
            self.id = player_id
        self.dominoes = dominoes

    @property
    def pieces_left(self) -> int:
        return len(self.dominoes)

    def get_highest_double(self) -> Optional[Domino]:
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
        """
        for domino in dominoes:
            self.remove_domino(domino)

    def __str__(self):
        return str({"id": self.id, "dominoes": self.dominoes})

    def __repr__(self):
        return str({"id": self.id, "dominoes": self.dominoes})


class Train:
    def __init__(
        self,
        dominoes: List[Domino] = [],
        player_id: Optional[str] = None,
        is_open: bool = False,
        train_id: Optional[str] = None,
    ):
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
        for domino in dominoes:
            self.add_domino(domino)

    def is_open_for_player(self, player_id: str) -> bool:
        return (
            (self.player_id is None) or (self.player_id == player_id) or (self.is_open)
        )

    def ends_in_double(self) -> bool:
        if len(self.dominoes) == 0:
            return False
        return is_double(self.dominoes[-1])

    def get_end_value(self) -> int:
        if len(self.dominoes) == 0:
            raise ValueError("Train has no dominoes")
        return self.dominoes[-1][1]

    def __str__(self):
        return str(
            {
                "id": self.id,
                "dominoes": self.dominoes,
                "is_open": self.is_open,
                "player_id": self.player_id,
            }
        )

    def __repr__(self):
        return str(
            {
                "id": self.id,
                "dominoes": self.dominoes,
                "is_open": self.is_open,
                "player_id": self.player_id,
            }
        )


class Board:
    def __init__(self, trains: List[Train] = [], engine: Optional[Domino] = None):
        self.trains = trains
        self.engine = engine
        self.contains_unfulfilled_double: bool = False
        self.unfulfilled_double_value: Optional[int] = None
        self.unfulfilled_double_train_id: Optional[str] = None

    def get_open_trains(self, player_id: str) -> List[Train]:
        return [train for train in self.trains if train.is_open_for_player(player_id)]

    def open_train(self, player: Player) -> None:
        for train in self.trains:
            if train.player_id == player.id:
                train.is_open = True

    def close_train(self, player: Player) -> None:
        for train in self.trains:
            if train.player_id == player.id:
                train.is_open = False

    def get_train_with_double(self) -> Optional[Train]:
        """
        Returns the train that ends in a double if one exists, otherwise
        returns None. Also updates the board's `contains_unfulfilled_double`
        and `unfulfilled_double_value` fields if necessary.
        """
        for train in self.trains:
            if train.ends_in_double():
                # these values should already be set correctly, so this is
                # likely redundant, but it's here just in case
                self.contains_unfulfilled_double = True
                self.unfulfilled_double_value = train.get_end_value()
                self.unfulfilled_double_train_id = train.id
                return train
        # these values should already be set correctly, so this is
        # likely redundant, but it's here just in case
        self.contains_unfulfilled_double = False
        self.unfulfilled_double_value = None
        self.unfulfilled_double_train_id = None
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

        if not board_has_communal_train:
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
        """
        # print("player", player)
        continuations = self.get_continuations(player)
        # print("continuations", continuations)
        choices = self.make_starting_choices_for_player(continuations, player)
        # print("choices", choices)
        return choices

    def __str__(self):
        return str({"trains": self.trains, "engine": self.engine})

    def __repr__(self):
        return str({"trains": self.trains, "engine": self.engine})


class MexicanTrain:
    def __init__(self, random_seed: Optional[int] = None) -> None:
        self.players: List[Player] = []
        self.board: Board = Board(trains=[], engine=None)
        self.is_first: bool = False

        self.turn: int = 0
        self.turn_count: int = 0
        self.player_count: int = 0
        self.player_agents: List[MexicanTrainBot] = []
        self.random_seed: Optional[int] = random_seed
        self.dominoes: List[Domino] = make_all_dominoes()

    def add_player(self, agent_class: "MexicanTrainBot") -> None:
        self.player_count += 1
        self.player_agents.append(agent_class)
        self.players.append(Player(dominoes=[], player_id=agent_class.name))

    # Up to 4 players take 15 dominoes each, 5 or 6 take 12 each, 7 or 8 take 10 each.
    def get_hand_size(self) -> int:
        if self.player_count < 5:
            return 15
        elif self.player_count < 7:
            return 12
        else:
            return 10

    def deal(self) -> None:
        random.seed(self.random_seed)
        random.shuffle(self.dominoes)

        hand_size = self.get_hand_size()
        for i in range(self.player_count):
            hand = self.dominoes[:hand_size]
            self.dominoes = self.dominoes[hand_size:]
            self.players[i].dominoes = hand

    def pickup(self, player: Player) -> Optional[Domino]:
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
        """
        # let the player pass if they're not obligated to fulfill a double
        if proposed_move is None or proposed_move.sequences_to_play is None:
            if self.board.contains_unfulfilled_double:
                if self.check_if_player_can_fulfill_double(player):
                    # they can't pass if they have a domino that can fulfill
                    # the double
                    return False
            return True

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

    def update_board_unfulfilled_double_status(
        self, move: Move, new_train_id: Optional[str]
    ) -> None:
        """
        Updates the board's `contains_unfulfilled_double`,
        `unfulfilled_double_value`, and `unfulfilled_double_train_id` fields
        based on the given move that was just made.
        """
        if move.ends_in_double:
            if move.sequences_to_play is None:
                raise Exception(
                    "Move can't end in double if it doesn't play any dominoes"
                )
            train_id_with_double = move.sequences_to_play[-1]["train_id"]
            if train_id_with_double is None:
                if new_train_id is None:
                    raise Exception("No train id provided for new train")
                train_id_with_double = new_train_id
            self.board.contains_unfulfilled_double = True
            self.board.unfulfilled_double_value = move.all_dominoes_played[-1][1]
            self.board.unfulfilled_double_train_id = train_id_with_double
        else:
            self.board.contains_unfulfilled_double = False
            self.board.unfulfilled_double_value = None
            self.board.unfulfilled_double_train_id = None

    def add_to_trains(self, player: Player, move: Move) -> str | None:
        """
        Adds the dominoes in the move to the trains specified by the move.
        If the move creates a new train, then the id of the new train is
        returned.

        Updates the board's `contains_unfulfilled_double`,
        `unfulfilled_double_value`, and `unfulfilled_double_train_id` fields
        if appropriate.
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
                    self.update_board_unfulfilled_double_status(
                        move, communal_mexican_train.id
                    )
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
                    self.update_board_unfulfilled_double_status(move, personal_train.id)
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
                        self.update_board_unfulfilled_double_status(move, None)
                        return

    def set_engine(self) -> bool:
        """
        Sets the first domino of the whole game (the train engine)
        equal to highest double of the first player who has at least
        one double. If no player has a double, returns False.
        """
        # Check for highest doubles
        for i in range(len(self.players)):
            player = self.players[i]
            highest_double = player.get_highest_double()
            if highest_double is not None:
                self.board.engine = highest_double
                player.remove_domino(highest_double)
                self.player_agents = shift_arr(self.player_agents, i)
                self.players = shift_arr(self.players, i)
                return True

        return False

    def win_condition(self):
        for player in self.players:
            if len(player.dominoes) == 0:
                return player.id
        return None

    def perform_move(self, player: Player, move: Move | None, is_first: bool) -> bool:
        """
        Performs the given move for the given player. Returns True if the
        player's turn is over, and False if the player gets to play again.
        """
        if move is None or move.sequences_to_play is None:
            self.pickup(player)
            self.board.open_train(player)
            # return False because the player gets to play again after picking
            # up a domino
            return False
        else:
            possible_new_train_id = self.add_to_trains(player, move)
            if move.ends_in_double:
                train_id_with_double = move.sequences_to_play[-1]["train_id"]
                if train_id_with_double is None:
                    if possible_new_train_id is None:
                        raise Exception("No train id provided for the new train")
                    train_id_with_double = possible_new_train_id
                double_value = move.all_dominoes_played[-1][1]
                new_domino = self.pickup(player)
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
        # Get agent move
        move = agent.play(player, self.board, self.is_first, piece_counts)

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
        move = agent.play(player, self.board, self.is_first, piece_counts)
        is_valid = self.check_valid_move(player, move, self.is_first)
        if not is_valid:
            raise Exception("Invalid move")
        if move is not None:
            self.perform_move(player, move, self.is_first)

    def play(self):
        self.deal()
        found_it = self.set_engine()

        if not found_it:
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
                print("Game over, no winner")
                return None

        print("The winner is player " + str(winner))
        return cur_player

    def __str__(self):
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


# Mexican Train Bot is a stub class that all player agents should inherit from
# it has a play method that takes in a player, board, is_first, and piece_counts
class MexicanTrainBot(ABC):
    def __init__(self, name: Optional[str]):
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
    ) -> Optional[Move]:
        pass


# This is a random agent that plays a random move
class RandomPlayerAgent(MexicanTrainBot):
    def play(
        self,
        player: Player,
        board: Board,
        is_first: bool,
        piece_counts: List[Tuple[str, int]],
    ) -> Optional[Move]:
        choices = board.get_choices(player)
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


for i in range(1000):
    mexican_train = MexicanTrain()
    mexican_train.add_player(RandomPlayerAgent("John"))
    mexican_train.add_player(RandomPlayerAgent("Jacob"))
    mexican_train.add_player(RandomPlayerAgent("Jingleheimer"))
    mexican_train.add_player(RandomPlayerAgent("Schmidt"))
    mexican_train.play()
