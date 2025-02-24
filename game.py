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
    # 2) Give the opportunity for players to buy the current card/cards on the discard pile
    # 3) Player either chooses to draw from the discard pile or draw from the top of the deck
    # 4) Player is either down / or not down. Player makes choice to go down or not go down
    # 5) If player is down, they can now choose to discard cards into other down piles(including their own)
    # 6) Player ends their turn discarding one card into the discard piles
    # 7) If the player has no cards in their hand at the end of the round they have won the game
    def game_flow(self):

        # 1) Identify which players turn it currenlty is
        current_player = self.players[self.current_turn]

        # Print output identifying who's turn it is
        print(f"\n{current_player.name}'s turn:")
        print(f"Hand: {current_player.get_hand()}")
        print(f"Discard card: {self.deck.peak_discard_card(1)}")

        # 2) Give the opportunity for players to buy the current card/cards on the discard pile
        #TODO: Need to have an action for others to choose if they are gonna buy but I will add that later.....

        # Define the order which players will get the opportunity to buy
        buy_order = []
        for i in range(1, 6) :
            buy_order.append((i + self.current_turn) % 6)

        # Give the opportunity for each player to buy if there are cards
        for buyer_id in buy_order:
            if self.deck.amount_in_discard() < 1 :
                continue
            buyer = self.players[buyer_id]
            
            # Show the player the cards available to buy
            while True :
                print(f"Discard card: {self.deck.peak_discard_card(3)}")
                buy = input(f"Does player {buyer.get_player_name()} want to buy, and how many cards?")
                
                # Don't allow them to buy more cards then are available...
                if buy[0] == 'b' :
                    amount = int(buy[1])
                    if amount > 0 and amount <= self.deck.amount_in_discard() :
                        for i in range(0, amount) :
                            buyer.draw_from_disc(self.deck)
                        break
                        
                        



            

        # 3) Player either chooses to draw from the discard pile or draw from the top of the deck
        game_control.player_draws_card_for_turn(current_player, self.deck)

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
        return True  




# Example usage:
game = Game(["Alice", "Bob", "Charlie", "David", "Eddie", "Fred"])
game.start_game()
while not game.is_game_over():
    game.next_turn()