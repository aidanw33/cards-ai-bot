from deck import Deck
from player import Player
from cards import Card
import game_control
from collections import Counter
import threading
import time
import gui

class Game:
    def __init__(self, player_names):
        self.deck = Deck()
        self.actions = ["d"]
        self.gui = None

        # Must be 6 players per game
        if len(player_names) != 6 :
            raise ValueError("There must be exactly six players per game")
        
        # Edit here if players can be AI
        self.players = [Player(name, False) for name in player_names] 

        # Deal the cards to the players to set up the hand 
        for player in self.players :
            player.draw(self.deck, 11) 

        #Place the top card in the discard pile
        self.deck.discard(self.deck.draw())

        # Determine the current turn, start on player 0
        self.current_turn = 0

    def get_game_state(self) : 
        
        # Returns the current game state in it's entirety
        game_state = {}
        # Get the linear one hot encoding for the discard pile  DECK 
        linear_encoding_discard_pile = self.deck.get_linear_encoding_discard_pile()
        game_state["linear_encoding_discard_pile"] = linear_encoding_discard_pile

        # Get the 2d matrix encoding for the discard pile for X cards DECK
        TWOd_matrix_encoding_discard_pile = self.deck.get_matrix_encoding_discard_pile(3)
        game_state["TWOd_matrix_encoding_discard_pile"] = TWOd_matrix_encoding_discard_pile

        # Get the one hot encoding for all cards in the draw pile DECK
        encoding_draw_pile = self.deck.get_encoding_deck()
        game_state["encoding_draw_pile"] = encoding_draw_pile

        # Get the one hot encoding for the players hands, both public and private hands PLAYER
        player_encodings_hand = []
        for player in self.players :
            player_encodings_hand.append(player.get_encoding_hand())
        game_state["player_encodings_hand"] = player_encodings_hand

        # Get the one hot encoding for the players down piles PLAYER
        player_encodings_down_pile = []
        for player in self.players :
            player_encodings_down_pile.append(player.get_encoding_down_pile())       
        game_state["player_encodings_down_pile"] = player_encodings_down_pile

        # Get the one hot encoding for cards we have not seen yet GAME

        # Get the one hot encoding for cards we have seen GAME 

        # Get the one hot encoding for all cards the an agent can see is in another players hands, public hand GAME

        # Get the turn of the current player
        game_state["current_player_turn"] = self.current_turn

        return game_state

    # Represents the game_flow of the game which can be broken down into a few different steps:
    # 1) Identify which players turn it currently is
    # 2) Give the opportunity for players to buy the current card/cards on the discard pile if current player doesn't want discard buy
    # 4) Player is either down / or not down. Player makes choice to go down or not go down
    # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
    # 6) Player ends their turn discarding one card into the discard piles
    # 7) If the player has no cards in their hand at the end of the round they have won the game
    def game_flow(self):

        while self.is_game_over() == False:

            # 1) Identify which players turn it currently is
            current_player = self.players[self.current_turn]

            # Print output identifying who's turn it is
            print(f"\n{current_player.get_player_name()}'s turn:")
            print(f"Hand: {current_player.get_hand()}")
            print(f"Discard card: {self.deck.peak_discard_card(1)}")


            # 2) Give the opportunity for players to buy the current card/cards on the discard pile if current player doesn't want discard buy
            game_control.player_draws_card_for_turn(current_player, self.deck, self.players, self.current_turn)

            # 4) Player is either down / or not down. Player makes choice to go down or not go down, must state which cards they are going down with
            game_control.player_decides_to_go_down_or_not(current_player)
                

            # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
            game_control.player_discards_into_down_piles(current_player, self.players, 1)

            # 6) Player ends their turn discarding one card into the discard piles
            game_control.player_discards_card_into_discard_pile(current_player, self.deck)

            # 7) If the player has no cards in their hand at the end of the round they have won the game
            if len(current_player.get_hand()) == 0:
                print(f"{current_player.get_player_name()} has won the game!")
                break

            self.current_turn = (self.current_turn + 1) % len(self.players)




    def start_game(self, starting_cards=5):
        
        start = input("Press enter to start game")
        
        gui.open_game_window()
        game_state = self.get_game_state()

        gui.set_images(game_state["player_encodings_hand"], game_state["TWOd_matrix_encoding_discard_pile"])

        # Begin the game
        self.game_flow()
        

    def is_game_over(self):
        # Define game-ending condition
        return False  

        

# Example usage:
game = Game(["Alice", "Bob", "Charlie", "David", "Eddie", "Fred"])

game.start_game()
while not game.is_game_over():
    game.next_turn()