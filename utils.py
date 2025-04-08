import numpy as np


def clean_field(field1, field2):
    """
    Perform a bitwise AND operation between two fields.
    
    Args:
        field1 (list): The first field represented as a 2D list.
        field2 (list): The second field represented as a 2D list.
        
    Returns:
        list: A new field that is the result of the bitwise AND operation.
    """
    field1_array = np.array(field1, dtype=np.int8)
    field2_array = np.array(field2, dtype=np.int8)
    return (field1_array & field2_array).tolist()


def fake_place(x, y, coordinates, field):
    """
    Simulate placing a piece on the field and calculate the resulting score.
    
    Args:
        x (int): The x-coordinate for placement.
        y (int): The y-coordinate for placement.
        coordinates (list): List of coordinate offsets defining the piece shape.
        field (list): The current game field as a 2D list.
        
    Returns:
        tuple: A tuple containing (score, updated_field) where score is the points earned
              and updated_field is the new state of the field after placement.
              Returns (0, original_field) if placement is invalid.
    """
    field_copy = np.array(field, dtype=np.int8)

    coords_x = [x + c[0] for c in coordinates]
    coords_y = [y + c[1] for c in coordinates]

    if (
        min(coords_x) < 0
        or max(coords_x) >= 10
        or min(coords_y) < 0
        or max(coords_y) >= 10
    ):
        return 0, field

    if any(field_copy[coords_y[i]][coords_x[i]] == 1 for i in range(len(coordinates))):
        return 0, field

    for i in range(len(coordinates)):
        field_copy[coords_y[i]][coords_x[i]] = 1

    lines, columns = [], []

    for line in range(10):
        if np.all(field_copy[line] == 1):
            lines.append(line)
            field_copy[line] = 0

    for col in range(10):
        if np.all(field_copy[:, col] == 1):
            columns.append(col)
            field_copy[:, col] = 0

    score = len(coordinates) + len(lines) * 10 + len(columns) * 10

    return score, field_copy.tolist()


def check_fake_lines(field):
    """
    Check for completed lines (rows) in the field and clear them.
    
    Args:
        field (list): The game field as a 2D list.
        
    Returns:
        tuple: A tuple containing (completed_lines, updated_field) where completed_lines
               is a list of indices of completed rows and updated_field is the field
               after clearing those rows.
    """
    field_np = np.array(field)
    lines = []

    for line in range(10):
        if np.all(field_np[line] == 1):
            lines.append(line)
            field_np[line] = 0

    return lines, field_np.tolist()


def check_fake_columns(field):
    """
    Check for completed columns in the field and clear them.
    
    Args:
        field (list): The game field as a 2D list.
        
    Returns:
        tuple: A tuple containing (completed_columns, updated_field) where completed_columns
               is a list of indices of completed columns and updated_field is the field
               after clearing those columns.
    """
    field_np = np.array(field)
    columns = []

    for col in range(10):
        if np.all(field_np[:, col] == 1):
            columns.append(col)
            field_np[:, col] = 0

    return columns, field_np.tolist()


def set_fake_filled(x, y, full, fake_field):
    """
    Set a specific cell in the field to a given value.
    
    Args:
        x (int): The x-coordinate of the cell.
        y (int): The y-coordinate of the cell.
        full (int): The value to set (typically 0 or 1).
        fake_field (list): The game field as a 2D list.
        
    Returns:
        list: The updated field after setting the cell value.
    """
    fake_field[y][x] = full
    return fake_field


def fake_fits(x, y, coordinates, field):
    """
    Check if a piece can be placed at the specified position on the field.
    
    Args:
        x (int): The base x-coordinate for placement.
        y (int): The base y-coordinate for placement.
        coordinates (list): List of coordinate offsets defining the piece shape.
        field (list): The current game field as a 2D list.
        
    Returns:
        bool: True if the piece can be placed, False otherwise.
    """
    for dx, dy in coordinates:
        tx, ty = x + dx, y + dy
        if not (0 <= tx < 10 and 0 <= ty < 10):
            return False
        if field[ty][tx] == 1:
            return False
    return True


def valid_moves_for_block_on_field(block, field):
    """
    Find all valid positions where a block can be placed on the field.
    
    Args:
        block (Block): The block object to place, must have coord_array attribute.
        field (list): The current game field as a 2D list.
        
    Returns:
        list: List of valid moves, where each move is [x, y, block] representing
              the position and block that can be placed.
    """
    moves = []
    coords = block.coord_array

    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)

    for y in range(-min_y, 10 - max_y):
        for x in range(-min_x, 10 - max_x):
            if fake_fits(x, y, coords, field):
                moves.append([x, y, block])

    return moves
