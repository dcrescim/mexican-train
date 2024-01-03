import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mexican_train.MexicanTrainBot import MexicanTrainBot
from mexican_train.player import Player
from mexican_train.board import Board
from mexican_train.domino_types import GameLogEntry, is_double, canonical, Domino
from mexican_train.move import Move
from typing import List, Tuple, Optional, Callable
import random


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
