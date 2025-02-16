class Card:
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    
    def __init__(self, rank, suit=None):
        if rank == "Joker":
            self.rank = "Joker"
            self.suit = None
        elif suit in self.suits and rank in self.ranks:
            self.rank = rank
            self.suit = suit
        else:
            raise ValueError("Invalid card rank or suit")
    
    def __str__(self):
        return f"{self.rank} of {self.suit}" if self.suit else "Joker"
    
    def __repr__(self):
        return self.__str__()
