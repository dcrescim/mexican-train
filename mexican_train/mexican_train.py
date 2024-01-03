import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import random
from typing import Tuple, Optional, List
from mexican_train.domino_types import Domino, GameLogEntry, make_all_dominoes, canonical, is_double, shift_arr
from mexican_train.train import Train
from mexican_train.player import Player
from mexican_train.MexicanTrainBot import MexicanTrainBot
from mexican_train.move import Move
from mexican_train.board import Board


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
            player_remaining_dominoes = list(
                dominoes_before_move - dominoes_played)
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
                        train_ends_in_double_after_move = is_double(
                            append_dominoes[-1])
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
                        raise Exception(
                            "No train id provided for the new train")
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
            piece_counts = [(p.id, p.pieces_left()) for p in self.players]
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