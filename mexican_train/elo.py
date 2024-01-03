
from typing import List, Dict, Optional, Tuple
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
        new_bot1_rating = self.ratings[bot1] + \
            self.k * (actual_score - expected_score)
        new_bot2_rating = self.ratings[bot2] + \
            self.k * (expected_score - actual_score)
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
        sorted_ratings = sorted(self.ratings.items(),
                                key=lambda x: x[0], reverse=True)
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
