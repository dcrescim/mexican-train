import sys
import os

# Calculate the path to the parent directory and add it to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from mexican_train.MexicanTrainBot import MexicanTrainBot
from mexican_train.NormalPlayerAgent import NormalPersonPlayerAgent
from mexican_train.RandomPlayerAgent import RandomPlayerAgent
from mexican_train.mexican_train import MexicanTrain
from mexican_train.elo import EloRating
import random
from typing import List, Tuple, Dict
import time

start_time = time.time()

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
            [aggregate_scores[player.name]
                for player in randomly_sampled_players],
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
