import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

default_image = "PNG-cards-1.3/2_of_clubs.png"  # Default image to use when a file is missing
discard_pile_x = 1.05
discard_pile_y_start = 0.4
discard_pile_spacing = 0.05
draw_pile_size = .1

def open_game_window():
    global fig, ax, img_artists, image_filenames, discard_pile_images, discard_pile_artists
    
    # Define the number of sections and columns
    global num_sections, num_columns
    num_sections = 6
    num_columns = 18
    
    # List of default image filenames for each subsection
    image_filenames = [[default_image] * num_columns for _ in range(num_sections)]
    discard_pile_images = ["discard_pile_1.png", "discard_pile_2.png", "discard_pile_3.png"]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width to make space for draw and discard piles
    ax.set_xticks([])  # Remove ticks
    ax.set_yticks([])  # Remove ticks
    ax.set_frame_on(False)  # Remove the frame

    # Stretch the content to the edges
    ax.set_xlim(0, 1)  # Set x-axis limits to stretch content to the full width
    ax.set_ylim(0, 1)  # Set y-axis limits to stretch content to the full height

    # Define section widths and heights based on axis limits
    width = 1 / num_columns  # Width of each card image (now scaled to fit the full width)
    height = 1 / num_sections  # Height of each card section (now scaled to fit the full height)

    # Draw sections with image tiles
    img_artists = []
    for i in range(num_sections):
        row = []
        for j in range(num_columns):
            # Calculate the correct position for each card in the row
            x_min = j * width 
            x_max = (j + 1) * width
            y_min = i * height
            y_max = (i + 1) * height
            image = load_image(image_filenames[i][j])
            
            # Place the image with adjusted extents
            img = ax.imshow(image, extent=[x_min, x_max, y_min, y_max])
            row.append(img)
        img_artists.append(row)

    # Draw pile (single image to the right)
    draw_pile_x = 1.05  # Position slightly outside the main sections
    draw_pile_y = 0.75  # Adjust height position
    draw_pile_size = 0.1  # Size of the image
    draw_pile_image = load_image("PNG-cards-1.3/Blank-Playing-Card.png")
    ax.imshow(draw_pile_image, extent=[draw_pile_x, draw_pile_x + draw_pile_size, draw_pile_y, draw_pile_y + draw_pile_size])
    ax.text(draw_pile_x + draw_pile_size / 2, draw_pile_y - 0.05, "Draw Pile", ha='center', va='center', fontsize=10, color='black')
    
    # Discard pile (three stacked images to the right, now stored as artists)
    discard_pile_x = 1.05
    discard_pile_y_start = 0.4
    discard_pile_spacing = 0.05
    discard_pile_artists = []  # Clear the artists list before populating
    for i, discard_image in enumerate(discard_pile_images):
        img = load_image(discard_image)
        discard_artist = ax.imshow(img, extent=[discard_pile_x, discard_pile_x + draw_pile_size, 
                                               discard_pile_y_start - i * discard_pile_spacing, 
                                               discard_pile_y_start + draw_pile_size - i * discard_pile_spacing])
        discard_pile_artists.append(discard_artist)  # Store the artist
        
    ax.text(discard_pile_x + draw_pile_size / 2, discard_pile_y_start - 3 * discard_pile_spacing - 0.05, "Discard Pile", ha='center', va='center', fontsize=10, color='black')
    
    # Set limits
    ax.set_xlim(0, 1.2)  # Extend x limit to fit piles (you can adjust this for more space)
    ax.set_ylim(0, 1)

    # Show plot
    plt.ion()
    plt.tight_layout()
    plt.show()


def load_image(filename):
    try:
        return mpimg.imread(filename)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default image.")
        return mpimg.imread(default_image)

def set_images(image_list, discard_vector):
    global image_filenames
    global discard_pile_images

    image_list = get_image_name_from_vector(image_list)
    discard_pile_images = get_image_name_from_vector_draw_pile(discard_vector)
    if len(image_list) != num_sections * num_columns:
        raise ValueError("Image list must contain exactly {} elements".format(num_sections * num_columns))
    
    image_filenames = [image_list[i * num_columns:(i + 1) * num_columns] for i in range(num_sections)]
    update_plot()

def get_image_name_from_vector_draw_pile(vector) :
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
    for v in vector: 
        hand_list = []
        for i in range(len(v)) :
            if v[i] == 1 :
                card_name_index = i % 54
                hand_list.append(f"PNG-cards-1.3/{card_names[card_name_index]}")
        buffer = 1 - len(hand_list)
        for i in range(buffer) :
            hand_list.append(f"PNG-cards-1.3/Blank-Playing-Card.png")
        image_list.extend(hand_list)
    print(image_list)
    return image_list

def get_image_name_from_vector(vector) :
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
    for v in vector: 
        hand_list = []
        for i in range(len(v)) :
            if v[i] == 1 :
                card_name_index = i % 54
                hand_list.append(f"PNG-cards-1.3/{card_names[card_name_index]}")
        buffer = 18 - len(hand_list)
        for i in range(buffer) :
            hand_list.append(f"PNG-cards-1.3/Blank-Playing-Card.png")
        image_list.extend(hand_list)
    return image_list

def update_plot():
    global img_artists
    global discard_pile_images
    for i in range(num_sections):
        for j in range(num_columns):
            image = load_image(image_filenames[i][j])
            img_artists[i][j].set_data(image)
    
    # Update the discard pile images (using set_data)
    for i, discard_image in enumerate(discard_pile_images):
        if i < 3 :
            img = load_image(discard_image)
            discard_pile_artists[i].set_data(img)  # Update discard pile images
    
    plt.draw()
