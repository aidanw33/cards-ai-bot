import tkinter as tk
from PIL import Image, ImageTk

class CardGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Game")
        self.root.geometry("800x600")  # Set initial window size

        # Canvas to draw the game area
        self.canvas = tk.Canvas(self.root, bg="green", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Make canvas resize with the window

        # Control button to start game
        self.start_button = tk.Button(self.root, text="Start Game", command=self.start_game)
        self.start_button.pack(side=tk.BOTTOM)

        # Colors for each section (just for visualization)
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

        # Image vectors for each section (initial empty)
        self.image_vectors = [[] for _ in range(6)]

    def start_game(self):
        """Start the game logic."""
        self.canvas.create_text(400, 250, text="Game Started!", font=('Helvetica', 24), fill='white')

        # Example initial vectors for each section (can be dynamically provided)
        self.image_vectors = [
            ['image1.png', 'image2.png', 'image3.png', 'image4.png', 'image5.png', 'image6.png', 'image7.png', 'image8.png', 'image9.png', 'image10.png', 
             'image11.png', 'image12.png', 'image13.png', 'image14.png', 'image15.png', 'image16.png', 'image17.png', 'image18.png', 'image19.png'],
            ['image21.png', 'image22.png', 'image23.png', 'image24.png', 'image25.png', 'image26.png', 'image27.png', 'image28.png', 'image29.png', 'image30.png', 
             'image31.png', 'image32.png', 'image33.png', 'image34.png', 'image35.png', 'image36.png', 'image37.png', 'image38.png', 'image39.png', 'image40.png'],
            ['image41.png', 'image42.png', 'image43.png', 'image44.png', 'image45.png', 'image46.png', 'image47.png', 'image48.png', 'image49.png', 'image50.png', 
             'image51.png', 'image52.png', 'image53.png', 'image54.png', 'image55.png', 'image56.png', 'image57.png', 'image58.png', 'image59.png', 'image60.png'],
            ['image61.png', 'image62.png', 'image63.png', 'image64.png', 'image65.png', 'image66.png', 'image67.png', 'image68.png', 'image69.png', 'image70.png', 
             'image71.png', 'image72.png', 'image73.png', 'image74.png', 'image75.png', 'image76.png', 'image77.png', 'image78.png', 'image79.png', 'image80.png'],
            ['image81.png', 'image82.png', 'image83.png', 'image84.png', 'image85.png', 'image86.png', 'image87.png', 'image88.png', 'image89.png', 'image90.png', 
             'image91.png', 'image92.png', 'image93.png', 'image94.png', 'image95.png', 'image96.png', 'image97.png', 'image98.png', 'image99.png', 'image100.png'],
            ['image101.png', 'image102.png', 'image103.png', 'image104.png', 'image105.png', 'image106.png', 'image107.png', 'image108.png', 'image109.png', 'image110.png', 
             'image111.png', 'image112.png', 'image113.png', 'image114.png', 'image115.png', 'image116.png', 'image117.png', 'image118.png', 'image119.png', 'image120.png']
        ]

        # Divide the canvas into 6 sections and display images based on the section-specific vectors
        self.divide_canvas_into_areas()

    def divide_canvas_into_areas(self):
        """Divide the canvas into 2 columns and 3 rows, filling the window, then each section into 10x2 smaller sections."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        section_width = canvas_width // 2  # Divide the width into 2 columns
        section_height = canvas_height // 3  # Divide the height into 3 rows

        for i in range(6):
            row = i // 2  # Calculate row (0, 1, 2)
            col = i % 2  # Calculate column (0 or 1)

            x1 = col * section_width
            y1 = row * section_height
            x2 = (col + 1) * section_width
            y2 = (row + 1) * section_height

            # Draw a rectangle for each player with a different color
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2, fill=self.colors[i])

            # Optionally, label the sections for each player
            self.canvas.create_text((x1 + x2) // 2, (y1 + y2) // 2, text=f"Player {i+1}", font=('Helvetica', 12), fill='white')

            # Now divide this section into 10x2 smaller sections (vertical)
            self.divide_section_into_10x2(x1, y1, x2, y2, self.image_vectors[i])

    def divide_section_into_10x2(self, x1, y1, x2, y2, image_vector):
        """Divide each section into a 10x2 grid of smaller sections (vertical orientation)."""
        # Number of columns and rows for each section (vertical orientation)
        cols = 10
        rows = 2

        # Calculate the width and height of each smaller section
        section_width = (x2 - x1) // cols
        section_height = (y2 - y1) // rows

        for row in range(rows):
            for col in range(cols):
                small_x1 = x1 + col * section_width
                small_y1 = y1 + row * section_height
                small_x2 = small_x1 + section_width
                small_y2 = small_y1 + section_height

                # Draw the smaller section (each will have a unique color from the colors list)
                color = self.get_color_for_section(row * cols + col)
                self.canvas.create_rectangle(small_x1, small_y1, small_x2, small_y2, outline="black", width=1, fill=color)

                # Check if there is a valid image in the vector to show in the box
                image_index = row * cols + col
                if image_index < len(image_vector):
                    image_path = image_vector[image_index]
                    self.load_and_show_image(image_path, small_x1, small_y1, section_width, section_height)

    def load_and_show_image(self, image_path, x1, y1, section_width, section_height):
        """Load and display the image in the subsection."""
        try:
            img = Image.open(image_path)  # Open the image file
            img = img.resize((section_width, section_height), Image.Resampling.LANCZOS)  # Resize the image
            tk_img = ImageTk.PhotoImage(img)  # Convert image for Tkinter

            # Place the image inside the subsection
            self.canvas.create_image(x1 + section_width // 2, y1 + section_height // 2, image=tk_img)

            # Keep a reference to the image to prevent it from being garbage collected
            self.canvas.image = tk_img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def get_color_for_section(self, index):
        """Return a color based on the index for each of the smaller sections."""
        colors = ['pink', 'cyan', 'gray', 'lime', 'teal', 'magenta', 'indigo', 'violet', 'brown', 'beige', 
                  'lightgreen', 'lightblue', 'yellowgreen', 'peachpuff', 'lightpink', 'lightyellow', 'lightcoral', 'lightgoldenrod', 'lavender', 'orchid']
        return colors[index % len(colors)]

    def update_image_vectors(self, new_image_vectors):
        """Method to update image vectors for all sections."""
        print("HERERERERERERE")
        new_images = []
        for new_image_vector in new_image_vectors :
            new_images.append(self.get_image_name_from_vector(new_image_vector))



        # Update the image vectors for each section
        self.image_vectors = new_images
        # Clear the canvas
        self.canvas.delete("all")
        # Re-render the canvas with updated image vectors
        self.divide_canvas_into_areas()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

    def get_image_name_from_vector(self, vector) :
        card_names = [
        "2_of_diamonds.png", "2_of_hearts.png", "2_of_clubs.png", "2_of_spades.png", 
        "3_of_diamonds.png", "3_of_hearts.png", "3_of_clubs.png", "3_of_spades.png",  
        "4_of_diamonds.png", "4_of_hearts.png", "4_of_clubs.png", "4_of_spades.png",  
        "5_of_diamonds.png", "5_of_hearts.png", "5_of_clubs.png", "5_of_spades.png",  
        "6_of_diamonds.png", "6_of_hearts.png", "6_of_clubs.png", "6_of_spades.png",  
        "7_of_diamonds.png", "7_of_hearts.png", "7_of_clubs.png", "7_of_spades.png",  
        "8_of_diamonds.png", "8_of_hearts.png", "8_of_clubs.png", "8_of_spades.png",  
        "9_of_diamonds.png", "9_of_hearts.png", "9_of_clubs.png", "9_of_spades.png",  
        "10_of_diamonds.png", "10_of_hearts.png", "10_of_clubs.png", "10_of_spades.png",  
        "jack_of_diamonds.png", "jack_of_hearts.png", "jack_of_clubs.png", "jack_of_spades.png",  
        "queen_of_diamonds.png", "queen_of_hearts.png", "queen_of_clubs.png", "queen_of_spades.png",  
        "king_of_diamonds.png", "king_of_hearts.png", "king_of_clubs.png", "king_of_spades.png",  
        "ace_of_diamonds.png", "ace_of_hearts.png", "ace_of_clubs.png", "ace_of_spades.png",  
        "red_joker.png", "black_joker.png"  
        ]

        image_list = []
        for i in range(len(vector)) :
            if vector[i] == 1 :
                card_name_index = i % 54
                image_list.append(f"PNG-cards-1.3/{card_names[card_name_index]}")
        return image_list
        

# Example usage
root = tk.Tk()
gui = CardGameGUI(root)

# Start the game
gui.start_game()

