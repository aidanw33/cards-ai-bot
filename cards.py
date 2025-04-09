class Card:
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace", "Joker"]
    decks = [0, 1]

    def __init__(self, rank, suit, deck):
        if suit in self.suits and rank in self.ranks and deck in self.decks:
            self.rank = rank
            self.suit = suit
            self.deck = deck
        else:
            raise ValueError("Invalid card rank or suit")
    
    def __str__(self):
        return f"{self.rank} of {self.suit} of deck {self.deck}" 
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # Check if the other object is a Card instance and has the same rank and suit
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit and self.deck == other.deck
        return False

    @staticmethod
    def map_to_encoding(card) :
        card_suite = card.suit
        index = 0
        if card_suite == "Diamonds":
            index += 0
        elif card_suite == "Hearts":
            index += 1
        elif card_suite == "Clubs":    
            index += 2
        elif card_suite == "Spades":
            index += 3
        
        card_rank = card.rank
        if card_rank == "2":
            index += 0 
        elif card_rank == "3":
            index += 2
        elif card_rank == "4":
            index += 6
        elif card_rank == "5":
            index += 10
        elif card_rank == "6":
            index += 14
        elif card_rank == "7":
            index += 18
        elif card_rank == "8":
            index += 22
        elif card_rank == "9":
            index += 26
        elif card_rank == "10":
            index += 30
        elif card_rank == "Jack":
            index += 34
        elif card_rank == "Queen":
            index += 38
        elif card_rank == "King":
            index += 42
        elif card_rank == "Ace":
            index += 46
        elif card_rank == "Joker":
            index += 50

        card_deck = card.deck
        if card_deck == 0:
            index += 0
        elif card_deck == 1:
            index += 54

        encoding = [0] * 108
        encoding[index] = 1
        return encoding, index

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
            'j': 'Jack', 'q': 'Queen', 'k': 'King', 'a': 'Ace', 'x': 'Joker'
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