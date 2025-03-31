from deck import Deck
from player import Player
from cards import Card
import game_control
from collections import Counter
import threading
import time
import gui
import sys

class Game:
    def __init__(self):
        self.deck = Deck()
        self.actions = ["d"]
        self.gui = None
        self.total_turns = 0
        self.is_game_over = False
        self.action_draw_disc = 0
        self.print = False
        # Must be 6 players per game
        '''
        if len(player_names) != 6 :
            raise ValueError("There must be exactly six players per game")'
        '''
        
        # Edit here if players can be AI
        self.players = [Player(name, False) for name in ["Alice", "Bob"]] 

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

        # Get the top card in the discard pile 
        if len(self.deck.discard_pile) > 0 :
            top_card_discard_pile = self.deck.peak_discard_card(1)[0]
            game_state["top_card_discard_pile"] = Card.map_to_encoding(top_card_discard_pile)[0]
        else :
            top_card_discard_pile = [0] * 108
            game_state["top_card_discard_pile"] = top_card_discard_pile

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

        # Get the encoding for ranks we have seen in the game so far
        encoding_player_down_ranks = [0] * 13
        
        for player in self.players :
            if player.get_is_player_down() :
                ranks = player.get_ranks_in_down_pile() 
                for rank in ranks :
                    if rank != "Jack" and rank != "Queen" and rank != "King" and rank != "Ace" :
                        rank = int(rank) - 1
                        encoding_player_down_ranks[rank] = 1
                    elif rank == "Jack" :
                        encoding_player_down_ranks[10] = 1
                    elif rank == "Queen" :
                        encoding_player_down_ranks[11] = 1
                    elif rank == "King" :
                        encoding_player_down_ranks[12] = 1
                    elif rank == "Ace" :
                        encoding_player_down_ranks[0] = 1

        game_state["encoding_player_down_ranks"] = encoding_player_down_ranks

        # Get the one hot encoding for cards we have not seen yet GAME

        # Get the one hot encoding for cards we have seen GAME 

        # Get the one hot encoding for all cards the an agent can see is in another players hands, public hand GAME

        # Get the turn of the current player
        game_state["current_player_turn"] = self.current_turn

        # Get the players amount of buys left
        game_state["buys_used"] = []
        for player in self.players :
            game_state["buys_used"].append(player.get_buys_used())        

        return game_state

    # Represents the game_flow of the game which can be broken down into a few different steps:
    # 1) Identify which players turn it currently is
    # 2) Give the opportunity for players to buy the current card/cards on the discard pile if current player doesn't want discard buy
    # 4) Player is either down / or not down. Player makes choice to go down or not go down
    # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
    # 6) Player ends their turn discarding one card into the discard piles
    # 7) If the player has no cards in their hand at the end of the round they have won the game
    def game_flow(self):

        while True :

            # 1) Identify which players turn it currently is
            current_player = self.players[self.current_turn]

            # Print output identifying who's turn it is
            if self.print :
                print(f"\n{current_player.get_player_name()}'s turn:")
                print(f"Hand: {current_player.get_hand()}")
                print(f"Discard card: {self.deck.peak_discard_card(1)}")


            # 2) Give the opportunity for players to buy the current card/cards on the discard pile if current player doesn't want discard buy
            game_control.player_draws_card_for_turn(current_player, self.deck, self.players, self.current_turn, self.get_game_state(), self.action_draw_disc)

            # 4) Player is either down / or not down. Player makes choice to go down or not go down, must state which cards they are going down with
            game_control.player_decides_to_go_down_or_not(current_player)
                

            # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
            game_control.agent_discards_card_into_discard_pile(current_player, self.players, 1)
            
            # Removing this choice for the beginning learning algo

            # 6) Player ends their turn discarding one card into the discard piles
            game_control.agent_player_discards_card_into_discard_pile(current_player, self.get_game_state(), self.deck)

            # 7) If the player has no cards in their hand at the end of the round they have won the game
            if len(current_player.get_hand()) == 0:
                self.is_game_over = True
                if self.print :
                    print(f"{current_player.get_player_name()} has won the game!")

                    print(self.total_turns, "total turns")

                break

            self.current_turn = (self.current_turn + 1) % len(self.players)
            
            # stop running when we get back to player 0
            if self.current_turn == 0 :
                break

            self.total_turns += 1

    def get_rewards(self) :
        if not self.is_game_over :
            return [0, 0]
        else :
            return game_control.calculate_player_scores(self.players)

    def reset(self) :
        self.deck = Deck()
        self.actions = ["d"]
        self.gui = None
        self.total_turns = 0
        self.is_game_over = False
        self.action_draw_disc = 0

        # Must be 6 players per game
        '''
        if len(player_names) != 6 :
            raise ValueError("There must be exactly six players per game")'
        '''
        
        # Edit here if players can be AI
        self.players = [Player(name, False) for name in ["Alice", "Bob"]] 

        # Deal the cards to the players to set up the hand 
        for player in self.players :
            player.draw(self.deck, 11) 

        #Place the top card in the discard pile
        self.deck.discard(self.deck.draw())

        # Determine the current turn, start on player 0
        self.current_turn = 0


    def start_game(self):
        
        #input("Press enter to start game")
        

        #gui.open_game_window()
        #game_state = self.get_game_state()

        #gui.set_images(game_state["player_encodings_hand"], game_state["TWOd_matrix_encoding_discard_pile"])

        # Begin the game
        self.game_flow()

    def take_action(self, action, current_player_turn) :
        self.action_draw_disc = action
        self.game_flow()




# Start the game with Two players
game = Game()
game.take_action(0, 0)


