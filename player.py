from cards import Card
from deck import Deck

class Player:
    def __init__(self, name, isAI):
        self.name = name
        self.hand = []
        self.isAI = isAI
        self.isDown = False

    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.draw()
            if card:
                self.hand.append(card)

    def get_hand(self):
        return self.hand
    
    def draw_from_deck(self, deck) :
        new_card = deck.draw()
        self.hand.append(new_card)

    def draw_from_disc(self, deck) :
        new_card = deck.draw_from_discard() 
        self.hand.append(new_card)

    #Removes card from hand, and puts it into the discard pile. 
    def card_into_discard(self, deck_to_discard, card_to_discard):
        for card in self.hand :
            if card_to_discard == card :
                deck_to_discard.discard(card_to_discard)
                self.hand.remove(card)
                print("discarding")
                return
        
        raise ValueError("Card is not in players hand, can not discard!")
    
    def get_is_player_down() :
        return self.isDown
    
    def set_is_player_down(is_player_down) :
        self.isDown = is_player_down
    
    def isAI(self):
        return self.isAI