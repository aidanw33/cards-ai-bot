import pytest
from player import Player
from deck import Deck
from cards import Card
import game_control
import rules

@pytest.fixture
def fresh_data():
    p = Player("test_user", True)
    d = Deck()
    return p, d

def test_player_going_down_test1(fresh_data) :
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



def test_is_a_valid_card_test1() :
    assert Card.is_a_valid_card("2h")

def test_is_a_valid_card_test2() :
    assert Card.is_a_valid_card("3d")

def test_is_a_valid_card_test3() :
    assert Card.is_a_valid_card("4c")

def test_is_a_valid_card_test4() :
    assert Card.is_a_valid_card("5s")

def test_is_a_valid_card_test5() :
    assert Card.is_a_valid_card("jo")

def test_is_a_valid_card_test6() :
    assert not Card.is_a_valid_card("2j")

def test_can_player_go_down_round1_test1(fresh_data, monkeypatch) :
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
    inputs = iter(["y", "3h 3d 3c 4s 4h 4d"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    game_control.player_decides_to_go_down_or_not(p)


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
    assert not rules.can_player_go_down(p, 1)

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
    assert rules.can_player_go_down(p, 1)

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
    assert rules.can_player_go_down(p, 1)
        
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
    assert not rules.can_player_go_down(p, 1)