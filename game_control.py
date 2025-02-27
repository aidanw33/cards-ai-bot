from deck import Deck
from player import Player
from cards import Card
import re

# Controls the beginning of the turn for a player, they either choose to draw from the deck or the discard pile
# If the player chooses to draw from the discard pile, this step is complete
# If the player chooses to draw from the deck, they give the table the opportunity to buy the card/cards on the discard pile
def player_draws_card_for_turn(current_player, deck, players, current_turn) :

    action = None
    while action != "deck" and action != "disc" :
        action = input("Draw from deck or disc?: ")

    if action == "deck" :

        print("#### Buying phase ####")
        # Define the order which players will get the opportunity to buy
        buy_order = []
        for i in range(1, 6) :
            buy_order.append((i + current_turn) % 6)

        # Give the opportunity for each player to buy if there are cards
        for buyer_id in buy_order: 
            if deck.amount_in_discard() < 1 :
                continue
            buyer = players[buyer_id]
            
            # Show the player the cards available to buy
            while True :
                print(f"Discard card: {deck.peak_discard_card(3)}")
                
                # guaranteed to be greater than 1
                cards_in_discard = deck.amount_in_discard()
                pattern = rf"\b(b[1-{cards_in_discard}]|no)\b"
                buy = ""
                while not bool(re.fullmatch(pattern, buy)) :
                    buy = input(f"Does player {buyer.get_player_name()} want to buy, and how many cards?(b#/no)")
                
                # Don't allow them to buy more cards then are available...
                if buy[0] == 'b' :
                    amount = int(buy[1])
                    buyer.buy_card(deck, amount)
                    break
                else :
                    break
        
        print("#### End of buying phase ####")

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
            if handcard.rank == rank and handcard.suit == suit :
                #this is the card we want to discard
                current_player.card_into_discard(deck, handcard)
                print(current_player.get_hand(), "hand after discard of ", handcard)
                return

