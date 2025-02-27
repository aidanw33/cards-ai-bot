from cards import Card
from deck import Deck

class Player:
    def __init__(self, name, isAI):
        self._name = name
        self._hand = []
        self._isAI = isAI
        self._isDown = False
        self._buys_used = 0

    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.draw()
            if card:
                self._hand.append(card)

    # Draws card from deck and puts it into the players hand, now throw errors if there are no cards left in the deck
    def draw_from_deck(self, deck) :
        if deck.amount_in_deck() < 1 :
            #TODO: Implement shuffling of discard pile into deck
            raise ValueError("There are no cards in the deck to draw from!")
        new_card = deck.draw()
        self._hand.append(new_card)

    # Draws card from discard pile and puts it into the players hand, throws error if there are no cards in the discard pile
    def draw_from_disc(self, deck) :
        if deck.amount_in_discard() < 1 :
            raise ValueError("There are no cards in the discard pile to draw from!")
        new_card = deck.draw_from_discard() 
        self._hand.append(new_card)
    
    # This method will attempt to buy the 'amount' of cards from the discard pile
    # If the player does not have enough buys, it will throw an error
    # If there are not enough cards in the discard pile, it will throw an error
    # If there are not enough cards in the deck, it will shuffle the old discard pile into the deck, excluding the cards that are being bought
    def buy_card(self, deck, amount) :
        if self._buys_used + amount > 3 :
            raise ValueError("Player can not buy more than 3 cards per round!")
        
        if deck.amount_in_discard() < amount :
            raise ValueError(f"Not enough cards in the discard pile to buy {amount} cards!")
        
        for i in range(0, amount) :
            self.draw_from_disc(deck)
            self.draw_from_deck(deck)
        
        self._buys_used += amount


    #Removes card from hand, and puts it into the discard pile. 
    def card_into_discard(self, deck_to_discard, card_to_discard):
        for card in self._hand :
            if card_to_discard == card :
                deck_to_discard.discard(card_to_discard)
                self._hand.remove(card)
                print("discarding")
                return
        
        raise ValueError("Card is not in players hand, can not discard!")
    
    ### GETTERS ###
    def get_is_player_down() :
        return self._isDown
        
    def get_hand(self):
        return self._hand

    def get_player_name(self) :
        return self._name

    def get_buys_used(self) :
        return self._buys_used

    ### SETTERS ###

    def set_buys_used(self, buys_used) :
        self._buys_used = buys_used
    
    def set_is_player_down(is_player_down) :
        self._isDown = is_player_down
    
    def get_isAI(self):
        return self._isAI