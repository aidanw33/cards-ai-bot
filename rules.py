from player import Player
from collections import Counter
from cards import Card
print = False
# Either returns True or False depending on if the player can go down on the current round(round_number)
def can_player_go_down(player, round_number):
    # Round 1 involves two sets of 3 cards, a.k.a two sets of 3 of a kind
    if round_number == 1:
        # Step 1: Count the frequency of each rank, and the number of jokers
        rank_counts = Counter(card.rank for card in player.get_hand() if card.rank != 'Joker')
        joker_count = sum(1 for card in player.get_hand() if card.rank == 'Joker')
        
        # Step 2: Track triplets and their ranks
        triplet_ranks = []
        
        # First, add any natural triplets (exactly 3 cards of same rank)
        for rank, count in rank_counts.items():
            if count == 3:
                triplet_ranks.append(rank)
        
        # Step 3: Try to form additional triplets using jokers
        remaining_jokers = joker_count
        for rank, count in rank_counts.items():
            if count == 2 and remaining_jokers > 0 and rank not in triplet_ranks:
                triplet_ranks.append(rank)
                remaining_jokers -= 1
            elif count == 1 and remaining_jokers >= 2 and rank not in triplet_ranks:
                triplet_ranks.append(rank)
                remaining_jokers -= 2
            elif count == 0 and remaining_jokers >= 3 and rank not in triplet_ranks:
                triplet_ranks.append(rank)
                remaining_jokers -= 3
        
        # Step 4: Return True/False and the list of triplet ranks
        can_go_down = len(triplet_ranks) >= 2
        return can_go_down, triplet_ranks[:2] if can_go_down else triplet_ranks
    
# Takes in a String of 6 cards seperated by a space, returns a boolean whether they are valid cards to go down with, will also return a list of cards if they are valid
def valid_cards_to_go_down_with(player, cards, round_number) :

    # Parse the cards as a string of inputs, verify that there are 6 cards
    cards = cards.split()
    if len(cards) != 6 :
        if print:
            print("You must go down with 6 cards, retry input")
        return False, None
    
    # Check if the cards are valid cards
    if not all(Card.is_a_valid_card(card) for card in cards) :
        if print:
            print("One of the cards you entered is not a valid card, retry input")
        return False, None
    
    # Verify that the cards are in the players hand, create a list of such cards
    for card in cards:
        is_card_in_hand, hcard = player.is_card_in_player_hand(card)
        if is_card_in_hand == False :
            if print:
                print(f"{card} is not in your hand, retry input")
            return False, None

    
    return True, cards

