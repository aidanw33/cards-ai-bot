from deck import Deck
from player import Player
from cards import Card



def player_draws_card_for_turn(current_player, deck) :

    action = None
    while action != "deck" and action != "disc" :
        action = input("Draw from deck or disc?: ")
    
    if action == "deck" :
        current_player.draw_from_deck(deck)
    
    if action == "disc" :
        current_player.draw_from_disc(deck)
    
    print(f"current_player hand after drawing from {action}: {current_player.get_hand()}")

def player_discards_card_into_discard_pile(current_player, deck) :

    print(f"Please discard a card that is in your hand: {current_player.get_hand()}")
    validCard = False
    while True: 
        
        card = input("Card to discard: ")

        if card == "jo" :
            suit = "Joker"
            rank = "Joker"
        else :
            suit = Card.map_to_suit(card[1])
            rank = Card.map_to_card_value(card[0])

        for handcard in current_player.get_hand() :
            print(handcard.suit, handcard.rank)
            if handcard.rank == rank and handcard.suit == suit :
                #this is the card we want to discard
                current_player.card_into_discard(deck, handcard)
                print(current_player.get_hand(), "hand after discard of ", handcard)
                return

