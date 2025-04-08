from tkinter import Tk, Label, Canvas, PhotoImage, Button
from tkinter.ttk import Combobox
from game import Game
from ai import AI


class Main:
    """Main class for the 1010 game GUI interface."""

    def __init__(self, ai_type):
        """
        Initialize the main game window and components.

        Parameters:
            ai_type (str): The type of AI to use for playing the game
        """
        self.window = Tk()
        self.window.title("1010")
        self.window.geometry("600x750")
        self.window.configure(background="#474747")
        self.window.resizable(False, False)

        self.game = Game(self)
        self.game.gui = self
        self.ai_type = ai_type
        self.AI = AI(ai_type)
        self.last_x = None
        self.last_y = None
        self.last_preview = []
        self.move_processing = False

        self.points_label = Label(
            self.window, font=("Segoe UI Light", 24), bg="#474747", fg="lightgray"
        )
        self.points_label["text"] = "0"
        self.points_label.place(x=(300 - self.points_label.winfo_width() / 2), y=10)

        self.ai_label = Label(
            self.window,
            text="AI Type:",
            font=("Segoe UI Light", 12),
            bg="#474747",
            fg="lightgray",
        )
        self.ai_label.place(x=50, y=20)

        self.ai_dropdown = Combobox(
            self.window, state="readonly", font=("Segoe UI Light", 12), width=10
        )
        self.ai_dropdown["values"] = (
            "MCTS",
            "heuristic",
            "greedy",
        )
        self.ai_dropdown.current(self.get_ai_type_index(ai_type))
        self.ai_dropdown.place(x=120, y=20)
        self.ai_dropdown.bind("<<ComboboxSelected>>", self.change_ai_type)

        self.canvas = Canvas(
            self.window, width=500, height=500, bg="lightgray", highlightthickness=0
        )
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Leave>", self.remove_last_values)
        self.canvas.place(x=50, y=75)

        self.lose_img = PhotoImage(file="./resources/LoseScreenOverlay.gif")
        self.img = PhotoImage(file="./resources/DragAndDropOverlay.gif")
        self.bc_overlay = PhotoImage(file="./resources/BlockCanvasOverlay.gif")

        self.block_canvas = Canvas(
            self.window, width=500, height=125, bg="lightgray", highlightthickness=0
        )
        self.block_canvas.place(x=50, y=525 + 50 + 25)

        self.block_canvas.create_image(0, 0, image=self.bc_overlay, anchor="nw")
        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")

        self.game.generate_blocks()
        self.render_current_blocks()

        self.window.mainloop()

    def get_ai_type_index(self, ai_type):
        """
        Return the index of the AI type in the dropdown values list.

        Parameters:
            ai_type (str): The AI type name to find

        Returns:
            int: The index position in the dropdown list, defaults to 0 if not found
        """
        ai_types = {
            "mcts": 0,
            "heuristic": 1,
            "greedy": 2,
        }
        return ai_types.get(ai_type.lower(), 0)

    def change_ai_type(self, event):
        """
        Handler for AI type change in dropdown.

        Parameters:
            event: The ComboboxSelected event
        """
        selected_ai_type = self.ai_dropdown.get()

        if selected_ai_type != self.ai_type:
            self.ai_type = selected_ai_type
            self.restart_game_with_new_ai()

    def restart_game_with_new_ai(self):
        """
        Restart the game with the currently selected AI type.
        Resets the board, points, and initializes new blocks.
        """
        self.AI = AI(self.ai_type)

        import tkinter

        for block in self.game.current_blocks:
            if hasattr(block, "canvas") and block.canvas:
                try:
                    block.canvas.destroy()
                except tkinter.TclError:
                    # Handle case when widget is already destroyed
                    pass
                block.canvas = None

        for i in range(len(self.game.field)):
            for j in range(len(self.game.field[i])):
                self.game.field[i][j] = 0

        self.game.points = 0
        self.points_label["text"] = "0"

        for y in range(10):
            for x in range(10):
                self.game.field[y][x] = 0
                self.canvas.create_rectangle(
                    x * 50,
                    y * 50,
                    x * 50 + 50,
                    y * 50 + 50,
                    fill="lightgray",
                    outline="",
                )

        self.restore_grid(self.img_id)

        import sys

        if "config" in sys.modules:
            if hasattr(sys.modules["config"], "last_block"):
                sys.modules["config"].last_block = None

        for item in self.block_canvas.find_all():
            if self.block_canvas.type(item) != "image":
                self.block_canvas.delete(item)

        self.game.current_blocks = []
        self.game.generate_blocks()
        self.render_current_blocks()

    def canvas_click(self, event=None):
        """
        Handle click events on the game board canvas.
        Uses AI to determine moves and places blocks.

        Parameters:
            event: The mouse click event (optional)
        """
        if self.move_processing:
            return

        self.move_processing = True

        try:
            self.AI.play_once(fake_main=self)

            if not self.AI.current_move:
                return

            x, y, block = self.AI.current_move

            if block not in self.game.current_blocks or getattr(
                block, "is_used", False
            ):
                self.render_current_blocks()
                return

            self.game.mark_block_used(block)
            self.game.selected_block = block

            if x < 10 and y < 10:
                self.place(x, y, block.coord_array)

                block.destroy()
                self.game.selected_block = None

                all_used = all(
                    getattr(block, "is_used", False)
                    for block in self.game.current_blocks
                )
                if all_used:
                    self.game.current_blocks = []
                    self.game.generate_blocks()
                    self.render_current_blocks()

                self._clear_completed_rows_and_columns()

                if not self.game.is_action_possible():
                    GUILoseScreen(self.window, self.game, self.lose_img)
        finally:
            self.move_processing = False

    def _clear_completed_rows_and_columns(self):
        """
        Clear completed rows and columns with animation.
        Identifies completed lines/columns and animates their removal.
        """
        completed_lines = self.game.check_lines()
        completed_columns = self.game.check_columns()

        if not (completed_lines or completed_columns):
            return

        cells_to_clear = []

        for line in completed_lines:
            for i in range(10):
                cells_to_clear.append((i, line))

        for column in completed_columns:
            for i in range(10):
                if (column, i) not in cells_to_clear:
                    cells_to_clear.append((column, i))

        self._animate_clearing(cells_to_clear)

        for line in completed_lines:
            self.game.clear_line(line)
            for i in range(10):
                self.clear_rect_on_coordinates(i, line)

        for column in completed_columns:
            self.game.clear_column(column)
            for i in range(10):
                self.clear_rect_on_coordinates(column, i)

    def _animate_clearing(self, cells_to_clear):
        """
        Create a flash animation for cells being cleared.

        Parameters:
            cells_to_clear (list): List of (x,y) coordinates to animate
        """
        import time

        flash_colors = [
            "#FFFFFF",
            "#FFF176",
            "#FFEE58",
            "#FFEB3B",
            "#FDD835",
            "#FBC02D",
            "#F9A825",
            "#F57F17",
        ]

        cell_items = {}
        for x in range(10):
            for y in range(10):
                items = self.canvas.find_overlapping(
                    x * 50 + 1, y * 50 + 1, x * 50 + 49, y * 50 + 49
                )
                for item in items:
                    if self.canvas.type(item) == "rectangle":
                        cell_items[(x, y)] = item

        for color in flash_colors:
            for x, y in cells_to_clear:
                if (x, y) in cell_items:
                    self.canvas.delete(cell_items[(x, y)])
                new_rect = self.canvas.create_rectangle(
                    x * 50, y * 50, x * 50 + 50, y * 50 + 50, fill=color, outline=""
                )
                cell_items[(x, y)] = new_rect

            self.window.update()
            time.sleep(0.005)

    def place(self, x, y, coordinates):
        """
        Place a block on the game board at specified coordinates.

        Parameters:
            x (int): X-coordinate for placement
            y (int): Y-coordinate for placement
            coordinates (list): List of relative coordinates defining the block shape
        """
        colors = [
            "#FF5252",
            "#FF4081",
            "#7C4DFF",
            "#536DFE",
            "#03A9F4",
            "#00BCD4",
            "#009688",
            "#4CAF50",
            "#8BC34A",
            "#FFEB3B",
            "#FFC107",
            "#FF9800",
        ]
        from config import rgb, last_block
        import random

        if hasattr(self.game.selected_block, "color"):
            selected_color = self.game.selected_block.color
        else:
            selected_color = colors[2 if not rgb else random.randint(0, len(colors) - 1)]

        if last_block is not None:
            for index in range(0, len(last_block[2])):
                self.draw_rect_on_coordinates(
                    last_block[0] + last_block[2][index][0],
                    last_block[1] + last_block[2][index][1],
                    "#FFA726",
                )

        import sys

        this_module = sys.modules["config"]
        this_module.last_block = [x, y, coordinates]

        self._animate_placement(x, y, coordinates, selected_color)

        for index in range(0, len(coordinates)):
            self.game.set_field(x + coordinates[index][0], y + coordinates[index][1], 1)

    def _animate_placement(self, x, y, coordinates, color):
        """
        Create a 'pop' animation for placing blocks.

        Parameters:
            x (int): X-coordinate for placement
            y (int): Y-coordinate for placement
            coordinates (list): List of relative coordinates defining the block shape
            color (str): HEX color code for the block
        """
        import time

        animated_items = []

        for i in range(3):
            scale = 0.5 + (i * 0.25)

            for item in animated_items:
                self.canvas.delete(item)
            animated_items = []

            for index in range(len(coordinates)):
                cx = x + coordinates[index][0]
                cy = y + coordinates[index][1]

                items = self.canvas.find_overlapping(
                    cx * 50 + 1, cy * 50 + 1, cx * 50 + 49, cy * 50 + 49
                )
                for item in items:
                    if self.canvas.type(item) == "rectangle":
                        self.canvas.delete(item)

                padding = int(25 * (1 - scale))
                scaled_x = cx * 50 + padding
                scaled_y = cy * 50 + padding
                scaled_width = 50 - (padding * 2)
                scaled_height = 50 - (padding * 2)

                bg_rect = self.canvas.create_rectangle(
                    cx * 50,
                    cy * 50,
                    cx * 50 + 50,
                    cy * 50 + 50,
                    fill="lightgray",
                    outline="",
                )

                rect = self.canvas.create_rectangle(
                    scaled_x,
                    scaled_y,
                    scaled_x + scaled_width,
                    scaled_y + scaled_height,
                    fill=color,
                    outline="",
                )
                animated_items.extend([bg_rect, rect])

            self.window.update()
            time.sleep(0.02 / 3)

        for item in animated_items:
            self.canvas.delete(item)

        for index in range(len(coordinates)):
            self.draw_rect_on_coordinates(
                x + coordinates[index][0], y + coordinates[index][1], color
            )

    def remove_last_values(self, event):
        """
        Remove preview of last block placement when mouse leaves canvas.

        Parameters:
            event: The mouse leave event
        """
        self.last_x = None
        self.last_y = None
        for index in range(0, len(self.last_preview)):
            lx = self.last_preview[index][0]
            ly = self.last_preview[index][1]
            if self.game.field[ly][lx] == 0:
                self.draw_rect(
                    self.last_preview[index][0],
                    self.last_preview[index][1],
                    "lightgray",
                )

    def draw_rect_on_coordinates(self, x, y, color):
        """
        Draw a rectangle at specific grid coordinates with the given color.

        Parameters:
            x (int): X grid position
            y (int): Y grid position
            color (str): HEX color code for the rectangle
        """
        self.draw_rect(x, y, color)

    def clear_rect_on_coordinates(self, x, y):
        """
        Clear a rectangle at specific grid coordinates.

        Parameters:
            x (int): X grid position
            y (int): Y grid position
        """
        self.draw_rect(x, y, "lightgray")

    def draw_rect(self, x, y, color):
        """
        Draw a rectangle at the specified grid position with visual effects.

        Parameters:
            x (int): X grid position
            y (int): Y grid position
            color (str): HEX color code for the rectangle
        """
        x_pos = x * 50
        y_pos = y * 50

        items = self.canvas.find_overlapping(
            x_pos + 1, y_pos + 1, x_pos + 49, y_pos + 49
        )
        for item in items:
            if self.canvas.type(item) == "rectangle":
                self.canvas.delete(item)

        if color != "lightgray":
            self.canvas.create_rectangle(
                x_pos, y_pos, x_pos + 50, y_pos + 50, fill=color, outline=""
            )

            highlight = self.canvas.create_polygon(
                x_pos,
                y_pos,
                x_pos + 50,
                y_pos,
                x_pos,
                y_pos + 50,
                fill="",
                width=0,
                smooth=True,
            )
            self.canvas.itemconfig(highlight, stipple="gray25", fill="white")

            shadow = self.canvas.create_polygon(
                x_pos + 50,
                y_pos,
                x_pos + 50,
                y_pos + 50,
                x_pos,
                y_pos + 50,
                fill="",
                width=0,
                smooth=True,
            )
            self.canvas.itemconfig(shadow, stipple="gray25", fill="black")
        else:
            self.canvas.create_rectangle(
                x_pos, y_pos, x_pos + 50, y_pos + 50, fill=color, outline=""
            )

        self.restore_grid(self.img_id)

    def render_current_blocks(self):
        """
        Render the current available blocks on the block canvas.
        Handles creation, coloring and placement of block visuals.
        """
        colors = [
            "#FF5252",
            "#FF4081",
            "#7C4DFF",
            "#536DFE",
            "#03A9F4",
            "#00BCD4",
            "#009688",
            "#4CAF50",
            "#8BC34A",
            "#FFEB3B",
            "#FFC107",
            "#FF9800",
        ]
        import random
        from config import rgb
        import tkinter

        for block in self.game.current_blocks:
            if hasattr(block, "canvas") and block.canvas:
                try:
                    block.canvas.place_forget()
                    block.canvas.destroy()
                except tkinter.TclError:
                    # Handle case when widget is already destroyed
                    pass
                block.canvas = None

        self.block_canvas.delete("all")
        self.block_canvas.create_image(0, 0, image=self.bc_overlay, anchor="nw")

        for index in range(0, len(self.game.current_blocks)):
            block = self.game.current_blocks[index]

            c = block.get_block_canvas()

            if not hasattr(block, "color"):
                if not rgb:
                    block.color = colors[2]
                else:
                    block.color = colors[random.randint(0, len(colors) - 1)]

            try:
                for item in c.find_all():
                    if c.type(item) == "rectangle":
                        c.itemconfig(item, fill=block.color)

                c.place(
                    x=50 + 166 * (index + 1) - 83 - int(c["width"]) / 2,
                    y=75 + 500 + 25 + (62 - int(c["height"]) / 2),
                )
            except (tkinter.TclError, AttributeError):
                if hasattr(block, "canvas") and block.canvas:
                    try:
                        block.canvas.destroy()
                    except tkinter.TclError:
                        pass
                    block.canvas = None

                c = block.__create_block_canvas()
                block.canvas = c

                for item in c.find_all():
                    if c.type(item) == "rectangle":
                        c.itemconfig(item, fill=block.color)

                c.place(
                    x=50 + 166 * (index + 1) - 83 - int(c["width"]) / 2,
                    y=75 + 500 + 25 + (62 - int(c["height"]) / 2),
                )

    def restore_grid(self, img_id):
        """
        Restore the grid overlay on the game board.

        Parameters:
            img_id: ID of the previous grid image to replace
        """
        if img_id:
            self.canvas.delete(img_id)

        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")


class GUILoseScreen:
    """Class to display game over screen and handle restart."""

    def __init__(self, window, game, lose_img):
        """
        Initialize the game over screen.

        Parameters:
            window: The main game window
            game: The game instance
            lose_img: Image to display on the lose screen
        """
        self.window = window
        self.game = game
        self.main = game.gui

        self.canvas = Canvas(
            window, width=600, height=725, bg="#474747", highlightthickness=0
        )
        self.canvas.create_image(0, 0, image=lose_img, anchor="nw")

        score_text = f"Your Score: {game.points}"
        self.canvas.create_text(
            300, 320, text=score_text, fill="white", font=("Segoe UI Light", 36)
        )

        replay_button = Button(
            self.canvas,
            text="Play Again",
            font=("Segoe UI", 16),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            command=self.replay_game,
        )
        self.canvas.create_window(300, 400, window=replay_button)

        self.canvas.place(x=0, y=0)

    def replay_game(self):
        """
        Restart the game with the current AI type.
        Resets the board, points, and initializes new blocks.
        """
        self.canvas.destroy()

        import tkinter

        for block in self.game.current_blocks:
            if hasattr(block, "canvas") and block.canvas:
                try:
                    block.canvas.destroy()
                except tkinter.TclError:
                    # Handle case when widget is already destroyed
                    pass
                block.canvas = None

            if hasattr(block, "is_used"):
                delattr(block, "is_used")

        for i in range(len(self.game.field)):
            for j in range(len(self.game.field[i])):
                self.game.field[i][j] = 0

        self.game.points = 0
        self.game.gui.points_label["text"] = "0"

        for y in range(10):
            for x in range(10):
                self.game.field[y][x] = 0
                self.game.gui.canvas.create_rectangle(
                    x * 50,
                    y * 50,
                    x * 50 + 50,
                    y * 50 + 50,
                    fill="lightgray",
                    outline="",
                )

        self.game.gui.restore_grid(self.game.gui.img_id)

        import sys

        if "config" in sys.modules:
            if hasattr(sys.modules["config"], "last_block"):
                sys.modules["config"].last_block = None

        for item in self.game.gui.block_canvas.find_all():
            if self.game.gui.block_canvas.type(item) != "image":
                self.game.gui.block_canvas.delete(item)

        self.game.current_blocks = []
        self.game.generate_blocks()
        self.game.gui.render_current_blocks()
