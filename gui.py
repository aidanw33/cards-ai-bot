import tkinter as tk
from tkinter import messagebox, ttk
import random
from PIL import Image, ImageTk

class CardGameGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("6-Player Card Game")
        self.master.geometry("800x900")  # Adjusted window size

        # Create a deck of cards
        self.deck = self.create_deck()
        self.players = self.create_players()

        # Create and set up the main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the deck frame
        self.deck_frame = tk.Frame(self.main_frame, relief=tk.RAISED, borderwidth=2)
        self.deck_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Create the deck label
        self.deck_label = tk.Label(self.deck_frame, text="Deck", font=("Arial", 14))
        self.deck_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Create the deal button
        self.deal_button = tk.Button(self.deck_frame, text="Deal Cards", command=self.deal_cards)
        self.deal_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Create the play area frame
        self.play_area_frame = tk.Frame(self.main_frame, relief=tk.RAISED, borderwidth=2)
        self.play_area_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the play area label
        self.play_area_label = tk.Label(self.play_area_frame, text="Players' Hands", font=("Arial", 14))
        self.play_area_label.pack(pady=5)

        # Create a scrollable frame for displaying players' hands
        self.scroll_frame = ttk.Frame(self.play_area_frame)
        self.scroll_canvas = tk.Canvas(self.scroll_frame, width=750, height=800)
        self.scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.scroll_window = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.scroll_window, anchor="nw")

        self.scroll_window.bind("<Configure>", self.on_configure)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Load card images
        self.card_images = self.load_card_images()

    def on_configure(self, event):
        """Update the scroll region when the contents change."""
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def create_deck(self):
        """Create a standard deck of 52 cards."""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']
        deck = [{'suit': suit, 'rank': rank} for suit in suits for rank in ranks]
        random.shuffle(deck)
        return deck

    def create_players(self):
        """Create 6 players with empty hands."""
        return [{'name': f'Player {i+1}', 'hand': []} for i in range(6)]

    def load_card_images(self):
        """Load and resize card images from the specified directory."""
        card_images = {}
        for suit in ['hearts', 'diamonds', 'clubs', 'spades']:
            for rank in ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']:
                image_path = f"PNG-cards-1.3/{rank}_of_{suit}.png"
                try:
                    image = Image.open(image_path)
                    # Resize the image to 60x80 pixels
                    image = image.resize((60, 80), Image.LANCZOS)
                    card_images[f"{rank}_of_{suit}"] = ImageTk.PhotoImage(image)
                except FileNotFoundError:
                    print(f"Warning: Image not found for {rank} of {suit}")
        return card_images

    def deal_cards(self):
        """Deal 8 cards to each player."""
        if len(self.deck) < 48:  # 6 players * 8 cards each
            messagebox.showinfo("Not Enough Cards", "Not enough cards in the deck to deal to all players.")
            return

        for player in self.players:
            player['hand'] = [self.deck.pop() for _ in range(8)]

        self.display_hands()

    def display_hands(self):
        """Display all players' hands in the scrollable frame."""
        for widget in self.scroll_window.winfo_children():
            widget.destroy()

        y = 10
        hand_width = 750  # Adjusted to fit 8 cards
        hand_height = 100

        for i, player in enumerate(self.players):
            # Create a frame for the player's hand
            hand_frame = tk.Frame(self.scroll_window, relief=tk.RAISED, borderwidth=2, width=hand_width, height=hand_height)
            hand_frame.pack(pady=5)

            # Display player name
            player_label = tk.Label(hand_frame, text=player['name'], font=("Arial", 12))
            player_label.pack(pady=5)

            # Display cards in the hand
            card_frame = tk.Frame(hand_frame)
            card_frame.pack(pady=5)

            for j, card in enumerate(player['hand']):
                card_image = self.card_images.get(f"{card['rank']}_of_{card['suit']}")
                if card_image:
                    card_label = tk.Label(card_frame, image=card_image)
                    card_label.image = card_image  # Keep a reference to prevent garbage collection
                    card_label.grid(row=0, column=j, padx=2, pady=2)
                else:
                    print(f"Warning: Image not found for {card['rank']} of {card['suit']}")

            y += hand_height + 10

if __name__ == "__main__":
    root = tk.Tk()
    game = CardGameGUI(root)
    root.mainloop()