from tkinter import Canvas


class Block:
    def __init__(self, block_list_index, blocks, gui):
        """
        Initialize a block with its configuration and UI elements.
        
        Args:
            block_list_index (int): Index of the block in the block list
            blocks (BLOCKS): The blocks collection containing block configurations
            gui: The GUI instance this block belongs to
        """
        self.block_list_index = block_list_index
        self.coord_array = blocks.block_list[block_list_index]
        self.gui = gui
        self.window = gui.window
        self.height = 0
        self.width = 0
        self.width_neg = 0
        self.set_measurement()
        self.canvas = None
        self.__create_block_canvas()

    def set_measurement(self):
        """
        Calculate and set the dimensions of the block based on its coordinates.
        Updates the width, height, and negative width offset properties.
        """
        width_pos = 0
        width_neg = 0
        height = 0
        for index in range(0, len(self.coord_array)):
            x1 = self.coord_array[index][0] * 25
            y1 = self.coord_array[index][1] * 25

            if x1 >= 0:
                if x1 + 25 > width_pos:
                    width_pos = x1 + 25
            elif x1 * -1 > width_neg:
                width_neg = x1 * -1

            if y1 + 25 > height:
                height = y1 + 25
        self.height = height
        self.width = width_pos + width_neg
        self.width_neg = width_neg

    def get_block_canvas(self):
        """
        Get the canvas of this block, creating it if it doesn't exist.
        
        Returns:
            Canvas: The tkinter Canvas object representing this block
        """
        if not self.canvas:
            self.__create_block_canvas()
        return self.canvas

    def __create_block_canvas(self):
        """
        Create or recreate the canvas for this block.
        Draws the block on the canvas according to its coordinates.
        
        Returns:
            Canvas: The newly created canvas
        """
        if self.canvas:
            try:
                self.canvas.destroy()
            except Exception:
                pass

        canvas = Canvas(
            self.window,
            width=self.width,
            height=self.height,
            bg="lightgray",
            highlightthickness=0,
        )
        canvas.bind("<Button-1>", self.select_block)
        for index in range(0, len(self.coord_array)):
            x1 = self.coord_array[index][0] * 25
            y1 = self.coord_array[index][1] * 25
            canvas.create_rectangle(
                x1 + self.width_neg,
                y1,
                x1 + 25 + self.width_neg,
                y1 + 25,
                fill="orange",
                outline="",
            )

        self.canvas = canvas
        return canvas

    def select_block(self, event):
        """
        Handle block selection when the canvas is clicked.
        
        Args:
            event: The click event that triggered the selection
        """
        selected_block = self.gui.game.selected_block
        if selected_block is not None and selected_block is not self:
            selected_block.remove_outline()
        self.gui.game.selected_block = self
        self.canvas["highlightthickness"] = 1

    def remove_outline(self):
        """
        Remove the selection outline from this block.
        """
        self.canvas["highlightthickness"] = 0

    def destroy(self):
        """
        Destroy the canvas object and clean up references.
        """
        if self.canvas:
            try:
                self.canvas.destroy()
            except Exception:
                pass
            self.canvas = None


class BLOCKS:
    def __init__(self):
        """
        Initialize the collection of available block configurations.
        Each block is represented by a list of coordinate pairs.
        """
        self.block_list = [
            [[0, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [0, 1], [0, 2]],
            [[0, 0], [0, 1], [0, 2], [0, 3]],
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[1, 0], [1, 1], [0, 1]],
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]],
            [[0, 0], [1, 0]],
            [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]],
            [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
        ]


class SimpleBlock:
    def __init__(self, coord_array):
        """
        Simple block representation for AI simulation.
        
        Args:
            coord_array (list): List of coordinate pairs defining the block's shape
        """
        self.coord_array = coord_array
