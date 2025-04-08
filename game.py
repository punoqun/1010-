import random
from blocks import Block, BLOCKS


class Game:
    def __init__(self, gui):
        """
        Initialize the game with GUI and default values.
        
        Args:
            gui: The GUI interface for the game
        """
        self.gui = gui
        self.field = [[0 for _ in range(10)] for _ in range(10)]
        self.points = 0
        self.blocks = BLOCKS()
        self.current_blocks = []
        self.selected_block = None
        self.used_blocks = [False, False, False]

    def check_lines(self):
        """
        Check which lines are completely filled.
        
        Returns:
            list: Indices of completely filled lines
        """
        return [line for line in range(10) if all(self.field[line])]

    def check_columns(self):
        """
        Check which columns are completely filled.
        
        Returns:
            list: Indices of completely filled columns
        """
        return [
            column for column in range(10) if all(row[column] for row in self.field)
        ]

    def get_points(self):
        """
        Get the current score.
        
        Returns:
            int: Current points
        """
        return self.points

    def add_points(self, points):
        """
        Add points to the current score and update the UI.
        
        Args:
            points (int): Number of points to add
        """
        self.points += points
        self.gui.points_label["text"] = str(self.points)
        self.gui.points_label.place(
            x=(300 - self.gui.points_label.winfo_width() / 2), y=10
        )

    def clear_line(self, index):
        """
        Clear a complete line at the given index.
        
        Args:
            index (int): Line index to clear
        """
        for i in range(10):
            self.set_field(i, index, 0)

    def clear_column(self, index):
        """
        Clear a complete column at the given index.
        
        Args:
            index (int): Column index to clear
        """
        for i in range(10):
            self.set_field(index, i, 0)

    def set_field(self, x, y, full):
        """
        Set a cell in the field to the given value and update score.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            full (int): Value to set (0 or 1)
        """
        self.add_points(1)
        self.field[y][x] = full

    def generate_blocks(self):
        """
        Generate three random blocks for the player to use.
        Resets the used status of all blocks.
        """
        block_count = len(self.blocks.block_list)
        self.current_blocks = [
            Block(random.randint(0, block_count - 1), self.blocks, self.gui)
            for _ in range(3)
        ]
        self.used_blocks = [False, False, False]

        for block in self.current_blocks:
            if hasattr(block, "is_used"):
                delattr(block, "is_used")

    def mark_block_used(self, block):
        """
        Mark a block as used and unavailable for further placement.
        
        Args:
            block: Block object to mark as used
        """
        block.is_used = True

    def is_block_usable(self, block):
        """
        Check if a block is still available for use.
        
        Args:
            block: Block object to check
            
        Returns:
            bool: True if the block exists and hasn't been used
        """
        return block in self.current_blocks and not getattr(block, "is_used", False)

    def are_all_blocks_used(self):
        """
        Check if all current blocks have been used.
        
        Returns:
            bool: True if all blocks are used
        """
        return all(getattr(block, "is_used", False) for block in self.current_blocks)

    def should_refresh_blocks(self):
        """
        Determine if new blocks should be generated.
        
        Returns:
            bool: True if all blocks are used or no valid moves exist
        """
        return self.are_all_blocks_used() or not self.is_action_possible()

    def fits(self, x, y, coordinates):
        """
        Check if a block can be placed at a specific position.
        
        Args:
            x (int): X position to place the block
            y (int): Y position to place the block
            coordinates (list): List of relative coordinates for the block cells
            
        Returns:
            bool: True if the block fits at the specified position
        """
        for dx, dy in coordinates:
            tx, ty = x + dx, y + dy
            if not (0 <= tx < 10 and 0 <= ty < 10) or self.field[ty][tx] == 1:
                return False
        return True

    def is_action_possible(self):
        """
        Check if any unused blocks can be placed anywhere on the field.
        
        Returns:
            bool: True if at least one valid move exists
        """
        for y in range(10):
            for x in range(10):
                if any(
                    self.fits(x, y, block.coord_array)
                    for block in self.current_blocks
                    if not getattr(block, "is_used", False)
                ):
                    return True
        return False

    def valid_moves(self):
        """
        Get all possible moves with unused blocks.
        
        Returns:
            list: List of valid moves [x, y, block]
        """
        moves = []
        for y in range(10):
            for x in range(10):
                for block in self.current_blocks:
                    if not getattr(block, "is_used", False) and self.fits(
                        x, y, block.coord_array
                    ):
                        moves.append([x, y, block])
        return moves

    @staticmethod
    def valid_moves_on_field(field, blocks):
        """
        Find all valid moves for given blocks on a specific field.
        
        Args:
            field (list): 2D field to check moves on
            blocks (list): List of blocks to check
            
        Returns:
            list: List of valid moves [x, y, block]
        """
        from utils import fake_fits

        moves = []
        for y in range(10):
            for x in range(10):
                for block in blocks:
                    if fake_fits(x, y, block.coord_array, field):
                        moves.append([x, y, block])
        return moves

    def valid_moves_for_block(self, block):
        """
        Find all valid positions for a specific block.
        
        Args:
            block: Block to find valid moves for
            
        Returns:
            list: List of valid moves [x, y, block]
        """
        moves = []
        for y in range(10):
            for x in range(10):
                if self.fits(x, y, block.coord_array):
                    moves.append([x, y, block])
        return moves

    def copy_field(self):
        """
        Create a deep copy of the current field.
        
        Returns:
            list: Copy of the game field
        """
        return [row[:] for row in self.field]
