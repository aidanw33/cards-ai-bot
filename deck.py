from cards import Card
import random

class Deck:
    def __init__(self):
        # Creates and shuffles a double deck of cards
        self.cards = [Card(rank, suit, deck) for suit in Card.suits for rank in Card.ranks for deck in Card.decks]
        self.cards.remove(Card("2", "Spades", 0))
        self.cards.remove(Card("2", "Spades", 1))
        self.cards.remove(Card("2", "Clubs", 0))
        self.cards.remove(Card("2", "Clubs", 1))
        self.shuffle()

        self.discard_pile = []

    # Creates an encoding of all the cards currently in the deck
    def get_encoding_deck(self) :
        encoding = [0] * 108
        for card in self.cards :
            _, index = Card.map_to_encoding(card)
            encoding[index] = 1

        return encoding
    
    # Creates an encoding of all the cards currently in the discard pile
    def get_linear_encoding_discard_pile(self) :
        encoding = [0] * 108
        for card in self.discard_pile :
            _, index = Card.map_to_encoding(card)
            encoding[index] = 1

        return encoding
    
    # Creates an encoding of the top amount_to_peek cards in the discard pile, returns a list of invidual card encodings
    def get_matrix_encoding_discard_pile(self, amount_to_peek) :
        encoding = []

        # If there are less cards in the discard pile than amount_to_peek, we peak all the cards in the discard pile
        new_amount_to_peek = amount_to_peek
        if len(self.discard_pile) < amount_to_peek :
            new_amount_to_peek = len(self.discard_pile)

        for card in self.discard_pile[-new_amount_to_peek:][::-1] :
            card_encoding, _ = Card.map_to_encoding(card)
            encoding.append(card_encoding)

        # Add an empty array if there aren't any cards present
        for i in range(amount_to_peek - new_amount_to_peek) :
            encoding.append([0] * 108)

        return encoding
    
    def shuffle(self):
        random.shuffle(self.cards)        

    def discard(self, Card):
        self.discard_pile.append(Card)
    
    def draw(self):
        return self.cards.pop() if self.cards else IndexError("Run out of cards bud time to fix it.")

    def draw_from_discard(self) :
        return self.discard_pile.pop() if self.discard_pile else IndexError("Run out of cards bud time to fix it.")
    
    def amount_in_discard(self) :
        return len(self.discard_pile)
    
    def amount_in_deck(self) :
        return len(self.cards)
    
    def reset_deck(self) :
        # Resets the deck by removing all cards from the discard pile and putting them back into the deck
        self.cards += self.discard_pile
        self.discard_pile = []
        self.shuffle()
        self.discard(self.draw())
    
    # Peaks amount_to_peek unless there are less than amount_to_peek cards in the discard pile, in that case it peaks all the cards in the discard pile
    def peak_discard_card(self, amount_to_peek):
        if len(self.discard_pile) > amount_to_peek :
            amount_to_peek = len(self.discard_pile)
        return self.discard_pile[-amount_to_peek:][::-1]

    
    def __str__(self):
        return f"Cards in deck: {self.cards} \n Cards in discard, {self.discard_pile[-1]}"