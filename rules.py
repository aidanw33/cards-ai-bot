from player import Player
from collections import Counter
from cards import Card

# Either returns True or False depending on if the player can go down on the current round(round_number)
def can_player_go_down(player, round_number) :
    

    # Round 1 involves two sets of 3 cards, a.k.a two sets of 3 of a kind
    if round_number == 1 :
        # Step 1: Count the frequency of each rank, and the number of jokers
        rank_counts = Counter(card.rank for card in player.get_hand() if card.rank != 'Joker')
        joker_count = sum(1 for card in player.get_hand() if card.rank == 'Joker')
        
        # Step 2: Check how many ranks have exactly 3 cards (or can form a triplet with jokers)
        triplet_counts = [count for count in rank_counts.values() if count == 3]
        
        # Step 3: Try to form triplets using jokers
        for rank, count in rank_counts.items():
            if count == 2 and joker_count > 0:
                triplet_counts.append(rank)  # Form a triplet using 1 joker
                joker_count -= 1
            elif count == 1 and joker_count >= 2:
                triplet_counts.append(rank)  # Form a triplet using 2 jokers
                joker_count -= 2
            elif count == 0 and joker_count >= 3:
                triplet_counts.append(rank)
                joker_count -= 3
        
        # Step 4: Return True if there are exactly 2 triplets, otherwise False
        return len(triplet_counts) >= 2
    
# Takes in a String of 6 cards seperated by a space, returns a boolean whether they are valid cards to go down with, will also return a list of cards if they are valid
def valid_cards_to_go_down_with(player, cards, round_number) :

    # Parse the cards as a string of inputs, verify that there are 6 cards
    cards = cards.split()
    if len(cards) != 6 :
        print("You must go down with 6 cards, retry input")
        return False, None
    
    # Check if the cards are valid cards
    if not all(Card.is_a_valid_card(card) for card in cards) :
        print("One of the cards you entered is not a valid card, retry input")
        return False, None
    
    # Verify that the cards are in the players hand, create a list of such cards
    for card in cards:
        is_card_in_hand, hcard = player.is_card_in_player_hand(card)
        if is_card_in_hand == False :
            print(f"{card} is not in your hand, retry input")
            return False, None

    
    return True, cards

