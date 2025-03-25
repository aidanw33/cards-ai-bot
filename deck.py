from cards import Card
import random

class Deck:
    def __init__(self):
        # Creates and shuffles a double deck of cards
        self.cards = [Card(rank, suit) for suit in Card.suits for rank in Card.ranks]
        self.cards.extend([Card(rank, suit) for suit in Card.suits for rank in Card.ranks])
        self.cards.extend([Card("Joker"), Card("Joker")])
        self.cards.extend([Card("Joker"), Card("Joker")])
        self.shuffle()

        self.discard_pile = []


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
    
    # Peaks amount_to_peek unless there are less than amount_to_peek cards in the discard pile, in that case it peaks all the cards in the discard pile
    def peak_discard_card(self, amount_to_peek):
        if len(self.discard_pile) > amount_to_peek :
            amount_to_peek = len(self.discard_pile)
        return self.discard_pile[-amount_to_peek:][::-1]

    
    def __str__(self):
        return f"Cards in deck: {self.cards} \n Cards in discard, {self.discard_pile[-1]}"