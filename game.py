import random
import string
from typing import Optional, List, Tuple, Any

Domino = Tuple[int, int]
Continuation = Tuple[int, Optional[str]]
Move = Tuple[List[Domino], Optional[str]]


def canonical(domino: Domino) -> Domino:
    return (min(domino), max(domino))


def make_all_dominoes() -> List[Domino]:
    dominoes = []
    for i in range(0, 12):
        for j in range(i, 12):
            dominoes.append((i, j))
    return dominoes


def shift_arr(arr: List[Any], shift) -> List[Any]:
    return arr[shift:] + arr[:shift]


def random_string(length=12) -> str:
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


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
            if domino[0] == domino[1]:
                if highest_double is None or domino[0] > highest_double[0]:
                    highest_double = domino
        return highest_double

    def remove_domino(self, domino: Domino) -> None:
        if domino in self.dominoes:
            self.dominoes.remove(domino)
            return
        if (domino[1], domino[0]) in self.dominoes:
            self.dominoes.remove((domino[1], domino[0]))
            return
        raise ValueError("Tried to remove domino that didn't exist")

    def remove_dominoes(self, dominoes: List[Domino]) -> None:
        for domino in dominoes:
            self.remove_domino(domino)

    def __str__(self):
        return str({"id": self.id, "dominoes": self.dominoes})

    def __repr__(self):
        return str({"id": self.id, "dominoes": self.dominoes})


class Train:
    def __init__(self, dominoes=[], player_id=None, is_open=False):
        self.id = random_string()
        if len(dominoes) == 0:
            raise ValueError("Train must have at least one domino")
        self.dominoes = dominoes
        self.is_open = is_open
        self.player_id = player_id

    def add_domino(self, domino: Domino) -> None:
        last_domino = self.dominoes[-1]
        if last_domino[1] == domino[0]:
            self.dominoes.append(domino)
        elif last_domino[1] == domino[1]:
            self.dominoes.append((domino[1], domino[0]))
        else:
            raise ValueError(
                "Cannot add domino {} to train {}".format(domino, self.dominoes)
            )

    def add_dominoes(self, dominoes: List[Domino]) -> None:
        for domino in dominoes:
            self.add_domino(domino)

    def is_open_for_player(self, player_id):
        return self.player_id is None or self.player_id == player_id or self.is_open

    def ends_in_double(self) -> bool:
        if len(self.dominoes) == 0:
            return False
        return self.dominoes[-1][0] == self.dominoes[-1][1]

    def get_end_value(self) -> Continuation:
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
    def __init__(self, trains=[], engine=None):
        self.trains = trains
        self.engine = engine

    def get_open_trains(self, player_id) -> List[Train]:
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
        for train in self.trains:
            if train.ends_in_double():
                return train
        return None

    def make_starting_choices_for_player(
        self, continuations: List[Continuation], player: Player
    ) -> List[Move]:
        all_starter_choices = []
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
                    possible_starting_move = ([new_domino], train_id)
                    all_starter_choices.append(possible_starting_move)
        return all_starter_choices

    def get_continuations(self, player: Player) -> List[Continuation]:
        double_train = self.get_train_with_double()
        if double_train is not None:
            return [(double_train.get_end_value(), double_train.id)]

        choices = []
        has_train = False
        has_mexican_train = False
        for train in self.trains:
            if train.player_id == player.id:
                has_train = True
                choices.append((train.get_end_value(), train.id))
            if train.player_id is None:
                has_mexican_train = True
                choices.append((train.get_end_value(), train.id))
            if train.is_open:
                choices.append((train.get_end_value(), train.id))

        # Make a new train
        if not has_train or (has_train and not has_mexican_train):
            choices.append((self.engine[1], None))

        return list(set(choices))

    def get_choices(self, player: Player) -> List[Move]:
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
        self.player_agents: List[Any] = []
        self.random_seed: Optional[int] = random_seed
        self.dominoes: List[Domino] = make_all_dominoes()

    def add_player(self, agent_class) -> None:
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

    def check_valid_move(
        self, player: Player, move: Optional[Move], is_first: bool = False
    ) -> bool:
        # Todo there are cases where this isn't the case
        # Like where you are forced to play a double if you have one
        if move is None:
            return True

        choices = self.board.get_choices(player)

        (append_dominoes, train_id) = move
        if not self.valid_domino_sequence(append_dominoes):
            return False

        if len(append_dominoes) == 0:
            return False

        if len(append_dominoes) > 1 and not is_first:
            return False

        first_domino_move = append_dominoes[0]
        for choice_dominoes, choice_train_id in choices:
            first_domino_choice = choice_dominoes[0]
            if train_id == choice_train_id and canonical(
                first_domino_move
            ) == canonical(first_domino_choice):
                return True
        return False

    def add_to_train(self, player: Player, move: Move) -> None:
        (append_dominoes, train_id) = move

        # You wish to make a new Train
        # Will this new train be the Mexican Train? or a personal one?
        if train_id is None:
            for train in self.board.trains:
                # You have a train, so make the Mexican Train
                if train.player_id == player.id:
                    self.board.trains.append(
                        Train(dominoes=append_dominoes, player_id=None)
                    )
                    player.remove_dominoes(append_dominoes)
                    return
            # You don't have a train, so make a personal one
            self.board.trains.append(
                Train(dominoes=append_dominoes, player_id=player.id)
            )
            player.remove_dominoes(append_dominoes)

        for train in self.board.trains:
            if train.id == train_id:
                train.add_dominoes(append_dominoes)
                player.remove_dominoes(append_dominoes)
                if train.player_id == player.id:
                    self.board.close_train(player)
                return

    def set_engine(self) -> bool:
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
        self, player: Player, agent: Any, piece_counts: List[Tuple[str, int]]
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


class RandomPlayerAgent:
    def __init__(self):
        pass

    def play(
        self, player: Player, board: Board, is_first, piece_counts
    ) -> Optional[Move]:
        choices = board.get_choices(player)
        if len(choices) == 0:
            return None

        random.shuffle(choices)
        return choices[0]


for i in range(1000):
    mexican_train = MexicanTrain()
    mexican_train.add_player(RandomPlayerAgent())
    mexican_train.add_player(RandomPlayerAgent())
    mexican_train.add_player(RandomPlayerAgent())
    mexican_train.add_player(RandomPlayerAgent())
    mexican_train.play()
