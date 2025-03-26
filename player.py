from cards import Card
from deck import Deck
from collections import Counter
class Player:
    def __init__(self, name, isAI):
        self._name = name
        self._hand = []
        self._isAI = isAI
        self._isDown = False
        self._buys_used = 0
        self._downPile = []

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
    
    # Removes card from players hand
    def remove_card_from_hand(self, card):
        self._hand.remove(card)
    
    # Get the one hot encoding for a players hand
    def get_encoding_hand(self) :
        encoding = [0] * 108
        for card in self._hand :
            _, index = Card.map_to_encoding(card)
            encoding[index] = 1

        return encoding

    def get_encoding_down_pile(self) :
        encoding = [0] * 108
        for card in self._downPile :
            _, index = Card.map_to_encoding(card)
            encoding[index] = 1

        return encoding
            
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
    
    # Either returns True or False depending on if the player can go down on the current round(round_number)
    def can_player_go_down(self, round_number) :
        
        # Round 1 involves two sets of 3 cards, a.k.a two sets of 3 of a kind
        if round_number == 1 :
            # Step 1: Count the frequency of each rank, and the number of jokers
            rank_counts = Counter(card.rank for card in self._hand if card.rank != 'Joker')
            joker_count = sum(1 for card in self._hand if card.rank == 'Joker')
            
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

    def can_player_go_down_with_cards(self, cards, round_number) :
        
        # Map the cards in the array to the actual cards in the players hand
        accessed_index = [] # Need the accessed index to make sure we do not count the same card twice for a duplicate card in a players hand
        player_cards = []
        for card in cards:
            if card == "jo" :
                suit = "Joker"
                rank = "Joker"
            else :
                suit = Card.map_to_suit(card[1])
                rank = Card.map_to_card_value(card[0])

            for handcard in self._hand :
                if handcard.rank == rank and handcard.suit == suit :
                    if self._hand.index(handcard) in accessed_index :
                        continue
                    player_cards.append(handcard)
                    accessed_index.append(self._hand.index(handcard))


        # Round 1 involves two sets of 3 cards, a.k.a two sets of 3 of a kind
        if round_number == 1 :
            # Step 1: Count the frequency of each rank, and the number of jokers
            rank_counts = Counter(card.rank for card in player_cards if card.rank != 'Joker')
            joker_count = sum(1 for card in player_cards if card.rank == 'Joker')
            
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
        
    # Checks to see if a card given in format({rank}{suite}{count(amount of times that card is in player's hand)}) is in the players hand
    def is_card_in_player_hand_count(self, card) :
        if card == "jo" :
            suit = "Joker"
            rank = "Joker"
            count = int(card[2])
        else :
            suit = Card.map_to_suit(card[1])
            rank = Card.map_to_card_value(card[0])
            count = int(card[2])

        actual_count = 0
        for handcard in self._hand() :
            if handcard.rank == rank and handcard.suit == suit :
                actual_count += 1
        
        return actual_count == count
    
    # Checks to see if a card given in format({rank}{suit}) is in the player's hand
    # Returns whether the card is in the player's hand or not, and then the card if it is in the player's hand, else None
    def is_card_in_player_hand(self, card):
        if card == "jo":
            suit = "Joker"
            rank = "Joker"
        else:
            suit = Card.map_to_suit(card[1])
            rank = Card.map_to_card_value(card[0])

        # Loop through the hand to see if the card is present
        for handcard in self._hand:
            if handcard.rank == rank and handcard.suit == suit:
                return True, handcard
        return False, None
    
    # Add card to the down pile
    def add_card_to_down_pile(self, card):
        valid_card, card = self.is_card_in_player_hand(card)
        if valid_card: 
            self._hand.remove(card)
            self._downPile.append(card)
        else:
            raise ValueError("Card is not in players hand, can not add to down pile!")

    # Add card to the down pile from the opponent
    def add_card_to_down_pile_from_opponent(self, card) :
        self._downPile.append(card)


    ### GETTERS ###
    def get_is_player_down(self) :
        return self._isDown
        
    def get_hand(self):
        return self._hand

    def get_player_name(self) :
        return self._name

    def get_buys_used(self) :
        return self._buys_used
    
    def get_down_pile(self) :
        return self._downPile

    ### SETTERS ###

    def set_buys_used(self, buys_used) :
        self._buys_used = buys_used
    
    def set_is_player_down(self, is_player_down) :
        self._isDown = is_player_down
    
    def get_isAI(self):
        return self._isAI
    
    