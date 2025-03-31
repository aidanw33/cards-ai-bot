import random


class Agent:
    """
    This class represents the agent that will play the game.
    It will have a method to choose a card to play based on the game state.
    """

    def __init__(self):
        pass

    @staticmethod
    def dumb_deck_or_disc(game_state):
        """
        This function takes in the game state and returns a list of all the cards in the deck or discard pile.
        """
        # Analyze the gamestate 

        # Choose to either draw from the deck or discard pile
        # 1 to draw from the deck, 0 to draw from the discard pile
        draw_from_deck = random.randint(0, 1)
        return draw_from_deck

    @staticmethod
    def dumb_buy_choice(game_state) :
        """
        This function decides how many cards to buy from the discard pile.
        Cannot exceed more than 3 cards, or more cards then are in the discard pile
        """

        # Analyze the gamestate
        discard_pile = game_state["TWOd_matrix_encoding_discard_pile"]
        cards_available = 0
        for card in discard_pile :
            is_card = any(x != 0 for x in card)
            if is_card :
                cards_available += 1
        
        # Choose a random number of cards to buy between 0 and cards_available, and not exceeding 3 for the player
        print(game_state["buys_used"])

        # Just return 0 for now, no buys
        return 0

    @staticmethod
    def dumb_discard(game_state) :
        """
        This function decides which card to discard from the players hand.
        It will discard the first card it finds in the hand
        """
        # Analyze the gamestate


        # Choose the card at index 0 to discard
        card_to_discard = 0
        return card_to_discard