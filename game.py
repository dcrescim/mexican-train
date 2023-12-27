import random
import string
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

Domino = Tuple[int, int]
Continuation = Tuple[int, Optional[str]]
Move = Tuple[List[Domino], Optional[str]]


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


def is_double(domino: Domino) -> bool:
    """
    Returns True if the given domino is a double, i.e. if both sides
    have the same number of dots.
    """
    return domino[0] == domino[1]


class Player:
    def __init__(self, dominoes: List[Domino] = []):
        self.id = random_string(6)
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
    ):
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
    ) -> List[Move]:
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
        all_starter_choices: List[Move] = []
        for end_val, train_id in continuations:
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
                    possible_starting_move: Move = ([new_domino], train_id)
                    all_starter_choices.append(possible_starting_move)
        return all_starter_choices

    def get_continuations(self, player: Player) -> List[Continuation]:
        """
        Returns a list of all possible continuations (playable numbers,
        and the corresponding open train that ends in said number) for
        the player's current turn.
        """
        double_train = self.get_train_with_double()
        if double_train is not None:
            return [(double_train.get_end_value(), double_train.id)]

        choices: List[Continuation] = []
        player_has_train = False
        board_has_communal_train = False
        for train in self.trains:
            if train.player_id == player.id:
                player_has_train = True
                choices.append((train.get_end_value(), train.id))
            if train.player_id is None:
                board_has_communal_train = True
                choices.append((train.get_end_value(), train.id))
            if train.is_open:
                choices.append((train.get_end_value(), train.id))

        # Record whether the player can make a new train
        if not player_has_train or (player_has_train and not board_has_communal_train):
            if self.engine is None:
                raise Exception("No engine in the train")
            choices.append((self.engine[1], None))

        return list(set(choices))

    def get_choices(self, player: Player) -> List[Move]:
        """
        Returns a list of all possible starting dominos and corresponding
        train ids for the player's current turn.

        The player's actual move may be a list of dominoes if they are
        playing a double or if it is their first turn, but this method only
        returns the first domino of each possible move. Any valid move the
        player makes MUST either be in this list or start with a
        domino/train combination in this list.
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
        self.players.append(Player(dominoes=[]))

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

    def pickup(self, player: Player) -> None:
        if len(self.dominoes) == 0:
            # print("No more dominoes to pickup")
            return
        domino = self.dominoes.pop()
        player.dominoes.append(domino)

    def valid_domino_sequence(self, dominoes: List[Domino]) -> bool:
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
        if proposed_move is None:
            # add logic for checking if the player is allowed to pass based on
            # whether or not they have to fulfill a double
            # the player is allowed to pass if there's no obligation to fulfill
            # a double
            return True

        allowed_choices = self.board.get_choices(player)

        (proposed_dominoes, proposed_train_id) = proposed_move
        if not self.valid_domino_sequence(proposed_dominoes):
            return False

        if len(proposed_dominoes) == 0:
            return False

        if len(proposed_dominoes) > 1 and not is_first:
            return False

        first_domino_move = proposed_dominoes[0]
        for allowed_first_dominoes, open_train_id in allowed_choices:
            first_domino_choice = allowed_first_dominoes[0]
            if proposed_train_id == open_train_id and canonical(
                first_domino_move
            ) == canonical(first_domino_choice):
                return True
        return False

    def update_board_unfulfilled_double_status(
        self, move: Move, new_train_id: Optional[str]
    ) -> None:
        train_ends_in_double_after_move = is_double(move[0][-1])
        if train_ends_in_double_after_move:
            train_id_with_double = move[1]
            if train_id_with_double is None:
                if new_train_id is None:
                    raise Exception("No train id provided for new train")
                train_id_with_double = new_train_id
            self.board.contains_unfulfilled_double = True
            self.board.unfulfilled_double_value = move[0][-1][1]
            self.board.unfulfilled_double_train_id = train_id_with_double
        else:
            self.board.contains_unfulfilled_double = False
            self.board.unfulfilled_double_value = None
            self.board.unfulfilled_double_train_id = None

    def add_to_train(self, player: Player, move: Move) -> None:
        """
        Adds the dominoes in the move to the train specified by the move.

        Updates the board's `contains_unfulfilled_double`,
        `unfulfilled_double_value`, and `unfulfilled_double_train_id` fields
        if appropriate.
        """
        (append_dominoes, train_id) = move

        # You wish to make a new Train
        # Will this new train be the Mexican Train? or a personal one?
        if train_id is None:
            for train in self.board.trains:
                # You have a train, so make the Mexican Train
                if train.player_id == player.id:
                    communal_mexican_train = Train(
                        dominoes=append_dominoes, player_id=None
                    )
                    self.board.trains.append(communal_mexican_train)
                    player.remove_dominoes(append_dominoes)
                    self.update_board_unfulfilled_double_status(
                        move, communal_mexican_train.id
                    )
                    return
            # You don't have a train, so make a personal one
            personal_train = Train(dominoes=append_dominoes, player_id=player.id)
            self.board.trains.append(personal_train)
            player.remove_dominoes(append_dominoes)
            self.update_board_unfulfilled_double_status(move, personal_train.id)

        for train in self.board.trains:
            if train.id == train_id:
                train.add_dominoes(append_dominoes)
                player.remove_dominoes(append_dominoes)
                train_ends_in_double_after_move = is_double(move[0][-1])
                if train.player_id == player.id and not train_ends_in_double_after_move:
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

    def perform_move(self, player: Player, move: Optional[Move]) -> bool:
        if move is None:
            self.pickup(player)
            self.board.open_train(player)
            return False
        else:
            self.add_to_train(player, move)
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
            raise Exception("Invalid move")

        # Attempt to perform move
        turn_complete = self.perform_move(player, move)
        if turn_complete:
            return

        # If the user draws a domino, they get to play again
        move = agent.play(player, self.board, self.is_first, piece_counts)
        is_valid = self.check_valid_move(player, move, self.is_first)
        if not is_valid:
            raise Exception("Invalid move")
        if move is not None:
            self.perform_move(player, move)

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
        return choices[0]


for i in range(1000):
    mexican_train = MexicanTrain()
    mexican_train.add_player(RandomPlayerAgent("John"))
    mexican_train.add_player(RandomPlayerAgent("Jacob"))
    mexican_train.add_player(RandomPlayerAgent("Jingleheimer"))
    mexican_train.add_player(RandomPlayerAgent("Schmidt"))
    mexican_train.play()
