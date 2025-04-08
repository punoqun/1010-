import random
from blocks import Block, BLOCKS

class Game:
    def __init__(self, gui):
        self.gui = gui
        # Use list comprehension for cleaner initialization
        self.field = [[0 for _ in range(10)] for _ in range(10)]
        self.points = 0
        self.blocks = BLOCKS()
        self.current_blocks = []
        self.selected_block = None
        self.used_blocks = [False, False, False]  # Track which blocks have been used

    def check_lines(self):
        return [line for line in range(10) if all(self.field[line])]

    def check_columns(self):
        return [column for column in range(10) if all(row[column] for row in self.field)]

    def get_points(self):
        return self.points

    def add_points(self, points):
        self.points += points
        self.gui.points_label["text"] = str(self.points)
        self.gui.points_label.place(x=(300 - self.gui.points_label.winfo_width() / 2), y=10)

    def clear_line(self, index):
        for i in range(10):
            self.set_field(i, index, 0)

    def clear_column(self, index):
        for i in range(10):
            self.set_field(index, i, 0)

    # Fixed typo in method name
    def set_field(self, x, y, full):
        self.add_points(1)
        self.field[y][x] = full

    def generate_blocks(self):
        block_count = len(self.blocks.block_list)
        self.current_blocks = [Block(random.randint(0, block_count - 1), self.blocks, self.gui) for _ in range(3)]
        self.used_blocks = [False, False, False]  # Reset used blocks tracking
        
        # Ensure no blocks have is_used flag
        for block in self.current_blocks:
            if hasattr(block, 'is_used'):
                delattr(block, 'is_used')

    def mark_block_used(self, block):
        # Just mark the block instance itself as used with a property
        block.is_used = True

    def is_block_usable(self, block):
        # Check if block exists and isn't marked as used
        return block in self.current_blocks and not getattr(block, 'is_used', False)

    def are_all_blocks_used(self):
        # Check if all blocks have been used
        return all(getattr(block, 'is_used', False) for block in self.current_blocks)
    
    def should_refresh_blocks(self):
        # Only refresh blocks when all have been used or no valid moves exist
        return self.are_all_blocks_used() or not self.is_action_possible()

    def fits(self, x, y, coordinates):
        for dx, dy in coordinates:
            tx, ty = x + dx, y + dy
            if not (0 <= tx < 10 and 0 <= ty < 10) or self.field[ty][tx] == 1:
                return False
        return True

    def is_action_possible(self):
        # Check if any unused blocks can be placed anywhere on the field
        for y in range(10):
            for x in range(10):
                if any(self.fits(x, y, block.coord_array) for block in self.current_blocks 
                      if not getattr(block, 'is_used', False)):
                    return True
        return False

    def valid_moves(self):
        moves = []
        for y in range(10):
            for x in range(10):
                for block in self.current_blocks:
                    # Only consider blocks that haven't been used
                    if not getattr(block, 'is_used', False) and self.fits(x, y, block.coord_array):
                        moves.append([x, y, block])
        return moves

    @staticmethod
    def valid_moves_on_field(field, blocks):
        from utils import fake_fits
        moves = []
        for y in range(10):
            for x in range(10):
                for block in blocks:
                    if fake_fits(x, y, block.coord_array, field):
                        moves.append([x, y, block])
        return moves

    def valid_moves_for_block(self, block):
        moves = []
        for y in range(10):
            for x in range(10):
                if self.fits(x, y, block.coord_array):
                    moves.append([x, y, block])
        return moves

    def copy_field(self):
        return [row[:] for row in self.field]
