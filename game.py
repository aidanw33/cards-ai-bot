from deck import Deck
from player import Player

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


    def next_turn(self):

        player = self.players[self.current_turn]

        print(f"\n{player.name}'s turn:")
        print(f"Hand: {player.show_hand()}")
        print(f"Discard card: {self.deck.peak_discard_card()}")


        #TODO: Need to have an action for others to choose if they are gonna buy but I will add that later.....

        action = "unrealaction unrealcard"
        while action.split()[0] not in self.actions :
            action = input("Action:")
        
        if action.split()[0] == "d" :
            card_to_discard = action.split()[1]

        #TODO: Is action legal?    

        self.current_turn = (self.current_turn + 1) % len(self.players)

    def start_game(self, starting_cards=5):
        
        start = input("Press enter to start game")
        
        # Begin the game
        self.next_turn()


    def is_game_over(self):
        # Define game-ending condition
        return True  




# Example usage:
game = Game(["Alice", "Bob", "Charlie", "David", "Eddie", "Fred"])
game.start_game()
print(game.players[0].show_hand())
while not game.is_game_over():
    game.next_turn()