from collections import defaultdict

class EloRating:
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: initial_rating)
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A"""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_rating(self, player_a, player_b, actual_score):
        """
        Update ratings after a match
        actual_score: 1 for A wins, 0.5 for draw, 0 for A loses
        """
        # Get current ratings
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        
        # Calculate expected scores
        expected = self.expected_score(rating_a, rating_b)
        
        # Update ratings
        self.ratings[player_a] += self.k_factor * (actual_score - expected)
        self.ratings[player_b] += self.k_factor * ((1 - actual_score) - (1 - expected))
    
    def get_ratings(self):
        """Return current ratings as a sorted dictionary"""
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    # Initialize ELO system
    elo = EloRating(k_factor=32)
