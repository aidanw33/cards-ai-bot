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

def test_encoding_cards_test1() :
    c1 = Card("2", "Diamonds", 0)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[0] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test2() :
    c1 = Card("3", "Diamonds", 0)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[2] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test3() :
    c1 = Card("3", "Clubs", 0)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[4] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test4() :
    c1 = Card("6", "Spades", 0)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[17] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test5() :
    c1 = Card("Joker", "Spades", 0)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[53] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test6() :
    c1 = Card("Joker", "Spades", 1)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[107] = 1
    assert c1_encoding == correct_encoding

def test_encoding_cards_test7() :
    c1 = Card("Ace", "Hearts", 1)
    c1_encoding, *_ = Card.map_to_encoding(c1)
    correct_encoding = [0] * 108
    correct_encoding[101] = 1
    assert c1_encoding == correct_encoding

def test_deck_creation() :
    d = Deck()
    assert len(d.cards) == 108

def test_deck_creation_encoding() :
    d = Deck()
    encoding = d.get_encoding_deck()
    correct_encoding = [1] * 108
    assert encoding == correct_encoding

def test_deck_creation_encoding2() :
    d = Deck() 
    d.cards.remove(Card("2", "Hearts", 0))
    encoding = d.get_encoding_deck()
    correct_encoding = [1] * 108
    correct_encoding[1] = 0
    assert encoding == correct_encoding

def test_deck_creation_encoding3() :
    d = Deck()
    d.cards.remove(Card("6", "Hearts", 0))
    d.cards.remove(Card("3", "Hearts", 1))
    print(d.cards)
    encoding = d.get_encoding_deck()
    correct_encoding = [1] * 108
    correct_encoding[15] = 0
    correct_encoding[57] = 0
    print(encoding)
    assert encoding == correct_encoding

def test_discard_encoding1() :
    d = Deck() 
    d.cards.remove(Card("6", "Hearts", 0))
    d.cards.remove(Card("3", "Hearts", 1))
    d.discard_pile.append(Card("6", "Hearts", 0))
    d.discard_pile.append(Card("3", "Hearts", 1))
    encoding_discard = d.get_linear_encoding_discard_pile()
    correct_encoding_discard = [0] * 108
    correct_encoding_discard[15] = 1
    correct_encoding_discard[57] = 1
    assert encoding_discard == correct_encoding_discard
    
def test_discard_encoding_top_cards2() :
    d = Deck() 
    d.cards.remove(Card("6", "Hearts", 0))
    d.cards.remove(Card("3", "Hearts", 1))
    d.discard_pile.append(Card("6", "Hearts", 0))
    d.discard_pile.append(Card("3", "Hearts", 1))
    encoding_discard = d.get_matrix_encoding_discard_pile(2)
    correct_encoding_discard_1 = [0] * 108
    correct_encoding_discard_2 = [0] * 108
    correct_encoding_discard_1[15] = 1
    correct_encoding_discard_2[57] = 1
    assert encoding_discard == [correct_encoding_discard_2, correct_encoding_discard_1]

# Tests requesting more cards than are available
def test_discard_encoding_top_cards3() :
    d = Deck() 
    d.cards.remove(Card("6", "Hearts", 0))
    d.cards.remove(Card("3", "Hearts", 1))
    d.discard_pile.append(Card("6", "Hearts", 0))
    d.discard_pile.append(Card("3", "Hearts", 1))
    encoding_discard = d.get_matrix_encoding_discard_pile(5)
    correct_encoding_discard_1 = [0] * 108
    correct_encoding_discard_2 = [0] * 108
    empty_encoding = [0] * 108
    correct_encoding_discard_1[15] = 1
    correct_encoding_discard_2[57] = 1
    print(len(encoding_discard))
    assert encoding_discard == [correct_encoding_discard_2, correct_encoding_discard_1, empty_encoding, empty_encoding, empty_encoding]

def test_player_encoding_hand() :
    p = Player("j", False)
    c = Card("Joker", "Spades", 1)
    p._hand.append(c)
    encoding_hand = p.get_encoding_hand()
    correct_encoding = [0] * 108
    correct_encoding[107] = 1
    assert correct_encoding == encoding_hand 







'''
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
encoding_discard
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
    assert not rules.can_player_go_down(p, 1)'
'''