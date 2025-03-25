class Card:
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    
    def __init__(self, rank, suit=None):
        if rank == "Joker":
            self.rank = "Joker"
            self.suit = "Joker"
        elif suit in self.suits and rank in self.ranks:
            self.rank = rank
            self.suit = suit
        else:
            raise ValueError("Invalid card rank or suit")
    
    def __str__(self):
        return f"{self.rank} of {self.suit}" if self.suit else "Joker"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # Check if the other object is a Card instance and has the same rank and suit
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

    @staticmethod
    def map_to_suit(char):
        suit_mapping = {
            'h': 'Hearts',
            'd': 'Diamonds',
            'c': 'Clubs',
            's': 'Spades'
        }
        return suit_mapping.get(char, 'Invalid suit')

    @staticmethod
    def map_to_card_value(char):
        value_mapping = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 't': '10',
            'j': 'Jack', 'q': 'Queen', 'k': 'King', 'a': 'Ace'
        }
        
        return value_mapping.get(char, 'Invalid card value')
    
    # Verifies whether a two character string is a valid card or not, True if it does correctly map to a card, False otherwise
    @staticmethod
    def is_a_valid_card(card):
        if len(card) != 2:
            return False
        if card == "jo":
            return True
        if card[0] in "23456789tjqka" and card[1] in "hdcs":
            return True
        else :
            return False