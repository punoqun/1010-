from tkinter import Canvas

class Block:
    def __init__(self, block_list_index, blocks, gui):
        self.block_list_index = block_list_index
        self.coord_array = blocks.block_list[block_list_index]
        self.gui = gui
        self.window = gui.window
        self.height = 0
        self.width = 0
        self.width_neg = 0
        self.set_measurement()
        self.canvas = None  # Initialize as None, create on demand
        self.__create_block_canvas()  # Create the initial canvas

    def set_measurement(self):
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
                width_neg = (x1 * -1)

            if y1 + 25 > height:
                height = y1 + 25
        self.height = height
        self.width = width_pos + width_neg
        self.width_neg = width_neg

    def get_block_canvas(self):
        # If the canvas doesn't exist or has been destroyed, create a new one
        if not self.canvas:
            self.__create_block_canvas()
        return self.canvas

    def __create_block_canvas(self):
        # If a canvas already exists, destroy it properly
        if self.canvas:
            try:
                self.canvas.destroy()
            except:
                pass  # Handle case where canvas might already be destroyed
        
        canvas = Canvas(self.window, width=self.width, height=self.height, bg="lightgray", highlightthickness=0)
        canvas.bind("<Button-1>", self.select_block)
        for index in range(0, len(self.coord_array)):
            x1 = self.coord_array[index][0] * 25
            y1 = self.coord_array[index][1] * 25
            canvas.create_rectangle(x1 + self.width_neg, y1, x1 + 25 + self.width_neg, y1 + 25, 
                                    fill="orange", outline="")

        self.canvas = canvas
        return canvas

    def select_block(self, event):
        selected_block = self.gui.game.selected_block
        if selected_block is not None and selected_block is not self:
            selected_block.remove_outline()
        self.gui.game.selected_block = self
        self.canvas["highlightthickness"] = 1

    def remove_outline(self):
        self.canvas["highlightthickness"] = 0

    def destroy(self):
        if self.canvas:
            try:
                self.canvas.destroy()
            except:
                pass  # Handle case where canvas might already be destroyed
            self.canvas = None  # Clear the reference


class BLOCKS:  # enum
    def __init__(self):
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
    """Simple block class for simulation in AI"""
    def __init__(self, coord_array):
        self.coord_array = coord_array
