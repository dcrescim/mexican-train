import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mexican_train.MexicanTrainBot import MexicanTrainBot
from mexican_train.player import Player
from mexican_train.board import Board
from mexican_train.domino_types import GameLogEntry, is_double, canonical
from mexican_train.move import Move
from typing import List, Tuple, Optional
import random


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
