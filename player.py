
class Player:
    def __init__(self, name, isAI):
        self.name = name
        self.hand = []
        self.isAI = isAI

    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.draw()
            if card:
                self.hand.append(card)

    def show_hand(self):
        return self.hand
    
    def isAI(self):
        return self.isAI