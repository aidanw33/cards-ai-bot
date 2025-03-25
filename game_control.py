from deck import Deck
from player import Player
from cards import Card
import re
from collections import Counter

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

def player_decides_to_go_down_or_not(current_player) :

    if current_player.get_is_player_down() == False :
        # Player can check if they can go down 
        can_be_down = current_player.can_player_go_down(1)

        if can_be_down :
            going_down = " "
            while going_down != "y" and going_down != "n" :
                going_down = input("Do you want to go down? (y/n)")
            if going_down == "y" :
                while True :
                    cards_going_down = input("Which cards are you going down with?(card1 card2 card3 card4 card5 card6)/exit")
                    if cards_going_down == "exit" :
                        break

                    # Make sure there are 6 cards to go down with
                    cards_going_down = cards_going_down.split()
                    if len(cards_going_down) != 6 :
                        print("You must go down with 6 cards, retry input")
                        continue
                    
                    # Check if the cards are valid cards
                    if not all(Card.is_a_valid_card(card_going_down) for card_going_down in cards_going_down) :
                        print("One of the cards you entered is not a valid card, retry input")
                        continue
                    
                    # Check if the cards in the hand are eligible to go down with
                    if not current_player.can_player_go_down_with_cards(cards_going_down) :
                        print("The cards you entered are not eligible to go down with, retry input")
                        continue


                    # add the frequency of each card to the end of the card
                    card_counts = Counter(cards_going_down)
                    cards_going_down = [card_going_down + str(cards_going_down[card_going_down]) if cards_going_down[card_going_down] > 1 else card_going_down + '1' for card_going_down in cards_going_down]

                    all_valid_cards = True
                    # Check if the cards are in the players hand
                    for cards_going_down_card in cards_going_down :
                        if current_player.is_card_in_player_hand_count(cards_going_down_card) == False :
                            print(f"{cards_going_down_card} is not in your hand, retry input")
                            all_valid_cards = False
                    
                    if not all_valid_cards :
                        continue
                        

                    if all_valid_cards :
                        # Remove the cards from the players hand
                        for card in cards_going_down :
                            current_player.add_card_to_down_pile(card)
                        current_player.set_is_player_down(True)
                        break
            if going_down == "n" :
                # If the player is not going down we can just continue
                pass