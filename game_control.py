from deck import Deck
from player import Player
from cards import Card
import re
from collections import Counter
import rules
from agent import Agent

_print = False

# Controls the beginning of the turn for a player, they either choose to draw from the deck or the discard pile
# If the player chooses to draw from the discard pile, this step is complete
# If the player chooses to draw from the deck, they give the table the opportunity to buy the card/cards on the discard pile
def player_draws_card_for_turn(current_player, deck, players, current_turn, game_state, action, opp_amount_to_buy) :

    
    # Action for playier, agent will be given game state to decide whether to draw from the deck or discard pile
    # Input will be the game state, output will be either 0 or 1 which maps to deck or discard pile
    
    '''
    while action != "deck" and action != "disc" :
        action = input("Draw from deck or disc?: ")

    if action == "deck" :
    '''
    # Only choose a random choice if we are the second player, otherwise it's being fed via AI

    if action == 1 :
        
        if _print :
            print("#### Buying phase ####")
        # Define the order which players will get the opportunity to buy
        buy_order = []
        for i in range(1, len(players)) :
            buy_order.append((i + current_turn) % len(players))

        
        # Give the opportunity for each player to buy if there are cards
        for buyer_id in buy_order: 
            if deck.amount_in_discard() < 1 :
                continue
            buyer = players[buyer_id]
            amount_to_buy = Agent.dumb_buy_choice(game_state)
            if current_turn == 1 :
                amount_to_buy = opp_amount_to_buy

            
            if amount_to_buy == 0 :
                if _print :
                    print(f"Player {buyer.get_player_name()} chose not to buy any cards")
                continue
            
            else :
                buyer.buy_card(deck, amount_to_buy)
                if _print :
                    print(f"Player {buyer.get_player_name()} bought {amount_to_buy} cards from the discard pile")


        '''
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
        '''
        current_player.draw_from_deck(deck)
    
    else :
        current_player.draw_from_disc(deck)
    
    if _print:
        print(f"current_player hand after drawing from {"deck" if action else "discard pile"}: {current_player.get_hand()}")

def player_discards_card_into_discard_pile(current_player, deck) :

    if _print:  
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
                if _print:
                    print(current_player.get_hand(), "hand after discard of ", handcard)
                return
            
def agent_player_discards_card_into_discard_pile(current_player, game_state, deck) :

    # Agent will discard a card in their hand
    # Input will be the game state, output will be the card that is being discarded
    card_to_discard = Agent.dumb_discard(game_state)

    player_hand = current_player.get_hand()

    current_player.card_into_discard(deck, player_hand[card_to_discard])

def agent_player_discards_card_into_discard_pile_beta(current_player, game_state, deck, card) :

    # Agent will discard a card in their hand
    # Input will be the game state, output will be the card that is being discarded
    card_to_discard = card

    player_hand = current_player.get_hand()

    current_player.card_into_discard(deck, card)

def agent_discards_card_into_down_pile(current_player, players, round_number) :

    if not current_player.get_is_player_down() :
        if _print:
            print("Player is not down, cannot discard into other down piles")
        return
    
    # Player is currenlty down, need the ranks of the cards in the other down piles
    for opponent in players :
        # If player is down inspect down piles
        if opponent.get_is_player_down() :
            # get the ranks of the cards in opponents down pile
            ranks = opponent.get_ranks_in_down_pile()

            for card in current_player.get_hand()[:] :

                # Always must have one card to discard at the end of the round
                if len(current_player.get_hand()) < 2 :
                    return
                if card.rank in ranks :
                    opponent.add_card_to_down_pile_from_opponent(card)
                    current_player.remove_card_from_hand(card)
                    if _print:
                        print(f"Player {current_player.get_player_name()} discarded {card} into {opponent.get_player_name()}'s down pile")
    



        

def player_discards_into_down_piles(current_player, players, round_number) :

    # Check if the player is down
    if current_player.get_is_player_down() == False :
        return
    
    # Check the number of cards in our hand, we cannot discard into other hands with less than 2 cards
    if len(current_player.get_hand()) < 2 :

        if _print :
            print("You cannot discard into other down piles with less than 2 cards in your hand")
        return
    
    # If we are down, and have cards to discard, we can now choose to discard into other down piles
    if round_number == 1 :
        
        # Find the eligible ranks to discard into
        eligible_ranks = {}
        for player in players :
            if player.get_is_player_down() == True :
                for card in player.get_down_pile() :
                    if card.rank not in eligible_ranks :
                        eligible_ranks[card.rank] = [player]
                    else :
                        eligible_ranks[card.rank].append(player)
        
        # Print the eligible ranks to discard into
        if _print:
            print("Eligible ranks to discard into:", eligible_ranks.keys())

        # Choose which cards to discard
        while True :
            cards_to_down_piles = input("Which cards are you discarding into other down piles?(card1 card2 ....)/exit")
            if cards_to_down_piles == "exit" :
                if _print:
                    print("Exiting discard into other down piles phase")
                break
            
            # Parse the cards as a string of inputs, verify that there are 2 cards
            cards_to_down_piles = cards_to_down_piles.split()
            if len(cards_to_down_piles) >= len(current_player.get_hand()) :
                if _print:
                    print("You cannot discard more than 1 less than the amount of cards in your hand, retry input")
                continue
            
            # Check if the cards are valid cards
            if not all(Card.is_a_valid_card(card_to_down_piles) for card_to_down_piles in cards_to_down_piles) :
                if _print:
                    print("One of the cards you entered is not a valid card, retry input")
                continue
            
            # Verify that the cards are in the players hand, create a list of such cards
            for card_to_down_piles in cards_to_down_piles:
                is_card_in_hand, hcard = current_player.is_card_in_player_hand(card_to_down_piles)
                if is_card_in_hand == False :
                    if _print:
                        print(f"{card_to_down_piles} is not in your hand, retry input")
                    continue

            # Check if the cards are eligible to discard into other down piles
            if not all(card_to_discard.rank in eligible_ranks for card_to_discard in cards_to_down_piles) :
                if _print:
                    print("One of the cards you entered is not eligible to discard into other down piles, retry input")
                continue
            
            # Remove the cards from the players hand
            for card_to_down_piles in cards_to_down_piles :
                # Get the card in the players hand
                is_card_in_hand, hcard = current_player.is_card_in_player_hand(card_to_down_piles)

                if is_card_in_hand :
                    # Remove the card from the players hand
                    current_player.remove_card_from_hand(hcard)

                    # Discard the card into a random discard pile of an opponent
                    eligible_ranks[hcard.rank][0].add_card_to_down_pile_from_opponent(hcard)
            
            break

def calculate_player_scores(players) :

    # Calculate the player scores and return a dictionary of the scores of each player
    player_scores = [0] * len(players)

    five_point_cards = set(["2", "3", "4","5", "6", "7"])
    ten_point_cards = set(["8", "9", "10", "Jack", "Queen", "King"])
    twenty_point_cards = set(["Ace", "Joker"])

    for i, player in enumerate(players) :
        player_score = 0
        for card in player.get_hand() :
            if card.rank in five_point_cards :
                player_score -= 5
            elif card.rank in ten_point_cards :
                player_score -= 10
            elif card.rank in twenty_point_cards :
                player_score -= 20
        if player_score == 0 :
            player_scores[i] = 1
            player_scores[((i + 1) % 2)] = -1
            return player_scores
        
        player_scores[i] = player_score
    return player_scores

def player_goes_down_or_not(current_player) :
    # Identify if the current player is down
    if current_player.get_is_player_down() == True :
        return
    
    # If the current player isn't down, check whether or not they have the cards to be down
    can_be_down = rules.can_player_go_down(current_player, 1)

    # Print whether a player can go down or not
    if not can_be_down :
        if _print:
            print("Cards in your hand are not eligible to go down with")
        return

    if can_be_down :
        pass
        #Identify which cards are going down, will return a vector of ranks that can be used to go down
    return None

    # Player wants to go down, they can now choose which 6 cards they want to go down with
    while True :
        cards_going_down = input("Which cards are you going down with?(card1 card2 card3 card4 card5 card6)/exit")
        if cards_going_down == "exit" :
            _print("Exiting go down phase, you will not go down")
            break
        
        valid_cards, valid_cards_list = rules.valid_cards_to_go_down_with(current_player, cards_going_down, 1) 
        if valid_cards :
            for card in valid_cards_list :
                current_player.add_card_to_down_pile(card)
            current_player.set_is_player_down(True)
            break
        else :
            _print("Invalid cards, retry input")
            continue


def player_decides_to_go_down_or_not(current_player) :

    # If player is down continue
    if current_player.get_is_player_down() == True :
        return
    can_be_down, down_ranks = rules.can_player_go_down(current_player, 1)

    # We are just making players automatically go down
    if can_be_down : 
        if _print:
            print("Player is going down with ranks: ", down_ranks)
        # Player is going down, all cards in hand which match down_ranks or jokers will go into the down pile
        for card in current_player.get_hand()[:] :
            if card.rank in down_ranks or card.rank == "Joker" :
                current_player.add_card_to_down_pile(card)
        current_player.set_is_player_down(True)



    ''''
    # Identify if the current player is down
    if current_player.get_is_player_down() == False :


        # If the current player isn't down, check whether or not they have the cards to be down
        can_be_down = rules.can_player_go_down(current_player, 1)

        # Print whether a player can go down or not
        if not can_be_down :
            print("Cards in your hand are not eligible to go down with")
        else : 
            going_down = ""
            while going_down != "y" and going_down != "n" :
                going_down = input("Do you want to go down? (y/n)")


            # If player wants to go down, they can, if they do not want to go down continue
            if going_down == "n" :
                pass
            else :  

                # Player wants to go down, they can now choose which 6 cards they want to go down with
                while True :
                    cards_going_down = input("Which cards are you going down with?(card1 card2 card3 card4 card5 card6)/exit")
                    if cards_going_down == "exit" :
                        print("Exiting go down phase, you will not go down")
                        break
                    
                    valid_cards, valid_cards_list = rules.valid_cards_to_go_down_with(current_player, cards_going_down, 1) 
                    if valid_cards :
                        for card in valid_cards_list :
                            current_player.add_card_to_down_pile(card)
                        current_player.set_is_player_down(True)
                        break
                    else :
                        print("Invalid cards, retry input")
                        continue

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
                    if not current_player.can_player_go_down_with_cards(cards_going_down, 1) :
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
                        break'
    '''