from deck import Deck
from player import Player
from cards import Card
import game_control

class Game:
    def __init__(self, player_names):
        self.deck = Deck()
        self.actions = ["d"]
        
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

            # 4) Player is either down / or not down. Player makes choice to go down or not go down
            #TODO: Implement this step

            # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
            #TODO: Implement this step

            # 6) Player ends their turn discarding one card into the discard piles
            game_control.player_discards_card_into_discard_pile(current_player, self.deck)

            # 7) If the player has no cards in their hand at the end of the round they have won the game
            #TODO: Implement this step 

            self.current_turn = (self.current_turn + 1) % len(self.players)

    def start_game(self, starting_cards=5):
        
        start = input("Press enter to start game")
        
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