import pytest
from player import Player
from deck import Deck
from cards import Card

@pytest.fixture
def fresh_data():
    p = Player("test_user", True)
    d = Deck()
    return p, d
    
def test_can_player_go_down_round1_test1(fresh_data) :
    p, d = fresh_data
    c1 = Card("3", "Hearts")
    c2 = Card("3", "Diamonds")
    c3 = Card("3", "Clubs")
    c4 = Card("4", "Spades")
    c5 = Card("4", "Hearts")
    c6 = Card("4", "Diamonds")
    p._hand.append(c1)
    p._hand.append(c2)
    p._hand.append(c3)  
    p._hand.append(c4)
    p._hand.append(c5)
    p._hand.append(c6)
    assert p.can_player_go_down(1)
        
def test_can_player_go_down_round1_test2(fresh_data) :
    p, d = fresh_data
    c1 = Card("3", "Hearts")
    c2 = Card("3", "Diamonds")
    c3 = Card("3", "Clubs")
    c4 = Card("4", "Spades")
    c5 = Card("4", "Hearts")
    p._hand.append(c1)
    p._hand.append(c2)
    p._hand.append(c3)  
    p._hand.append(c4)
    p._hand.append(c5)
    assert not p.can_player_go_down(1)

def test_can_player_go_down_round1_test3(fresh_data) :
    p, d = fresh_data
    c1 = Card("3", "Hearts")
    c2 = Card("3", "Diamonds")
    c3 = Card("3", "Clubs")
    c4 = Card("4", "Spades")
    c5 = Card("4", "Hearts")
    c6 = Card("Joker", "Joker")
    p._hand.append(c1)
    p._hand.append(c2)
    p._hand.append(c3)  
    p._hand.append(c4)
    p._hand.append(c5)
    p._hand.append(c6)
    assert p.can_player_go_down(1)

def test_can_player_go_down_round1_test4(fresh_data) :
    p, d = fresh_data
    c1 = Card("3", "Hearts")
    c2 = Card("3", "Diamonds")
    c3 = Card("Joker", "Joker")
    c4 = Card("4", "Spades")
    c5 = Card("4", "Hearts")
    c6 = Card("Joker", "Joker")
    p._hand.append(c1)
    p._hand.append(c2)
    p._hand.append(c3)  
    p._hand.append(c4)
    p._hand.append(c5)
    p._hand.append(c6)
    assert p.can_player_go_down(1)
        
def test_can_player_go_down_round1_test5(fresh_data) :
    p, d = fresh_data
    c1 = Card("3", "Hearts")
    c2 = Card("3", "Diamonds")
    c3 = Card("3", "Clubs")
    c4 = Card("4", "Spades")
    c5 = Card("5", "Hearts")
    c6 = Card("Joker", "Joker")
    c7 = Card("7", "Clubs")
    p._hand.append(c1)
    p._hand.append(c2)
    p._hand.append(c3)  
    p._hand.append(c4)
    p._hand.append(c5)
    p._hand.append(c6)
    p._hand.append(c7)
    assert not p.can_player_go_down(1)