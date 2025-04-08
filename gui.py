from tkinter import Tk, Label, Canvas, PhotoImage, Button
from tkinter.ttk import Combobox  # Add Combobox for dropdown
from game import Game
from ai import AI

class Main:
    def __init__(self, ai_type):
        self.window = Tk()
        self.window.title("1010")
        self.window.geometry("600x750")
        self.window.configure(background='#474747')
        self.window.resizable(False, False)

        self.game = Game(self)
        self.game.gui = self  # Add reference to Main instance in Game
        self.ai_type = ai_type  # Store the AI type
        self.AI = AI(ai_type)
        self.last_x = None
        self.last_y = None
        self.last_preview = []
        self.move_processing = False  # Flag to prevent multiple clicks being processed at once

        # Points label
        self.points_label = Label(self.window, font=("Segoe UI Light", 24), bg="#474747", fg="lightgray")
        self.points_label["text"] = "0"
        self.points_label.place(x=(300 - self.points_label.winfo_width() / 2), y=10)

        # AI Type Selection Dropdown
        self.ai_label = Label(self.window, text="AI Type:", font=("Segoe UI Light", 12), bg="#474747", fg="lightgray")
        self.ai_label.place(x=50, y=20)
        
        self.ai_dropdown = Combobox(self.window, state="readonly", font=("Segoe UI Light", 12), width=10)
        self.ai_dropdown['values'] = ('MCTS', 'heuristic', 'greedy')  # Adjust based on your available AI types
        self.ai_dropdown.current(self.get_ai_type_index(ai_type))
        self.ai_dropdown.place(x=120, y=20)
        self.ai_dropdown.bind("<<ComboboxSelected>>", self.change_ai_type)

        self.canvas = Canvas(self.window, width=500, height=500, bg="lightgray", highlightthickness=0)
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Leave>", self.remove_last_values)
        self.canvas.place(x=50, y=75)

        self.lose_img = PhotoImage(file='./resources/LoseScreenOverlay.gif')
        self.img = PhotoImage(file='./resources/DragAndDropOverlay.gif')
        self.bc_overlay = PhotoImage(file='./resources/BlockCanvasOverlay.gif')

        self.block_canvas = Canvas(self.window, width=500, height=125, bg="lightgray", highlightthickness=0)
        self.block_canvas.place(x=50, y=525 + 50 + 25)

        self.block_canvas.create_image(0, 0, image=self.bc_overlay, anchor="nw")
        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")

        self.game.generate_blocks()
        self.render_current_blocks()

        self.window.mainloop()

    def get_ai_type_index(self, ai_type):
        """Return the index of the AI type in the dropdown values list"""
        ai_types = {'random': 0, 'greedy': 1, 'advanced': 2}  # Adjust based on your AI types
        return ai_types.get(ai_type.lower(), 0)  # Default to first option if not found

    def change_ai_type(self, event):
        """Handler for AI type change in dropdown"""
        selected_ai_type = self.ai_dropdown.get()
        
        # Only restart if the selection actually changed
        if selected_ai_type != self.ai_type:
            self.ai_type = selected_ai_type
            self.restart_game_with_new_ai()
    
    def restart_game_with_new_ai(self):
        """Restart the game with the currently selected AI type"""
        # Create a new AI with the selected type
        self.AI = AI(self.ai_type)
        
        # Clean up existing blocks before creating new ones
        for block in self.game.current_blocks:
            if hasattr(block, 'canvas') and block.canvas:
                try:
                    block.canvas.destroy()
                except:
                    pass  # Handle case where canvas might already be destroyed
                block.canvas = None  # Clear the reference
        
        # Reset the game field
        for i in range(len(self.game.field)):
            for j in range(len(self.game.field[i])):
                self.game.field[i][j] = 0
        
        # Reset points
        self.game.points = 0
        self.points_label["text"] = "0"
        
        # Redraw the entire game board
        for y in range(10):
            for x in range(10):
                self.game.field[y][x] = 0
                self.canvas.create_rectangle(
                    x * 50, y * 50, 
                    x * 50 + 50, y * 50 + 50, 
                    fill="lightgray", outline=""
                )
        
        # Restore the grid overlay
        self.restore_grid(self.img_id)
        
        # Reset any global variables if needed
        import sys
        if 'config' in sys.modules:
            if hasattr(sys.modules['config'], 'last_block'):
                sys.modules['config'].last_block = None
        
        # Clear the block canvas before adding new blocks
        for item in self.block_canvas.find_all():
            if self.block_canvas.type(item) != "image":  # Preserve the background image
                self.block_canvas.delete(item)
        
        # Generate new blocks and render them
        self.game.current_blocks = []
        self.game.generate_blocks()
        self.render_current_blocks()

    def canvas_click(self, event=None):
        # Prevent multiple clicks from being processed simultaneously
        if self.move_processing:
            return
            
        self.move_processing = True
        
        try:
            # Get AI's move
            self.AI.play_once(fake_main=self)
            
            if not self.AI.current_move:
                return  # No valid move found
                
            x, y, block = self.AI.current_move
            
            # Safety check: ensure the block is actually in current_blocks and hasn't been used
            if block not in self.game.current_blocks or getattr(block, 'is_used', False):
                # The block was already removed or used - refresh display and return
                self.render_current_blocks()
                return
                
            # Mark this block as "in use" immediately to prevent double-selection
            self.game.mark_block_used(block)
            self.game.selected_block = block
            
            # Place block if valid position
            if x < 10 and y < 10:
                # Place the block and update the game state
                self.place(x, y, block.coord_array)
                
                # Hide the block visually but keep it in the array
                block.destroy()
                self.game.selected_block = None
                
                # Check if all blocks are used, then generate new ones
                all_used = all(getattr(block, 'is_used', False) for block in self.game.current_blocks)
                if all_used:
                    # Clear current blocks and generate new ones
                    self.game.current_blocks = []
                    self.game.generate_blocks()
                    self.render_current_blocks()

                # Clear completed lines and columns
                self._clear_completed_rows_and_columns()
                
                # Check if game is over
                if not self.game.is_action_possible():
                    GUILoseScreen(self.window, self.game, self.lose_img)
        finally:
            # Always reset the processing flag when done
            self.move_processing = False

    def _clear_completed_rows_and_columns(self):
        """Helper method to clear completed rows and columns with animation"""
        # Get completed lines and columns first
        completed_lines = self.game.check_lines()
        completed_columns = self.game.check_columns()
        
        if not (completed_lines or completed_columns):
            return
        
        # Create animation for cleared cells
        cells_to_clear = []
        
        # Collect cells from completed rows
        for line in completed_lines:
            for i in range(10):
                cells_to_clear.append((i, line))
        
        # Collect cells from completed columns
        for column in completed_columns:
            for i in range(10):
                # Avoid duplicates (intersections)
                if (column, i) not in cells_to_clear:
                    cells_to_clear.append((column, i))
        
        # Flash animation before clearing
        self._animate_clearing(cells_to_clear)
        
        # Actually clear the cells in the game data
        for line in completed_lines:
            self.game.clear_line(line)
            for i in range(10):
                self.clear_rect_on_coordinates(i, line)
        
        for column in completed_columns:
            self.game.clear_column(column)
            for i in range(10):
                self.clear_rect_on_coordinates(column, i)
    
    def _animate_clearing(self, cells_to_clear):
        """Create a simple flash animation for cells being cleared"""
        import time
        
        # Flash colors for animation
        flash_colors = ["#FFFFFF", "#FFF176", "#FFEE58", "#FFEB3B", "#FDD835", "#FBC02D", "#F9A825", "#F57F17"]
        
        # Get all existing canvas items and their coordinates
        cell_items = {}
        for x in range(10):
            for y in range(10):
                items = self.canvas.find_overlapping(x*50+1, y*50+1, x*50+49, y*50+49)
                for item in items:
                    if self.canvas.type(item) == "rectangle":
                        cell_items[(x, y)] = item
        
        # Animation loop
        for color in flash_colors:
            # Update all cells to be cleared with the current flash color
            for x, y in cells_to_clear:
                # Delete existing rectangle if it exists
                if (x, y) in cell_items:
                    self.canvas.delete(cell_items[(x, y)])
                # Create new rectangle with flash color
                new_rect = self.canvas.create_rectangle(
                    x*50, y*50, x*50+50, y*50+50,
                    fill=color, outline=""
                )
                cell_items[(x, y)] = new_rect
            
            # Update the display
            self.window.update()
            time.sleep(0.005)  # Keep animation speed fast
        
        # Final cleanup - we'll let the calling method handle the final state

    def place(self, x, y, coordinates):
        # Enhanced color palette with more vibrant options
        colors = [
            "#FF5252",   # Bright red
            "#FF4081",   # Pink
            "#7C4DFF",   # Deep purple
            "#536DFE",   # Indigo
            "#03A9F4",   # Light blue
            "#00BCD4",   # Cyan
            "#009688",   # Teal
            "#4CAF50",   # Green
            "#8BC34A",   # Light green
            "#FFEB3B",   # Yellow
            "#FFC107",   # Amber
            "#FF9800"    # Orange
        ]
        from config import rgb, last_block
        import random
        
        # Use the block's existing color instead of generating a new one
        if hasattr(self.game.selected_block, 'color'):
            selected_color = self.game.selected_block.color
        else:
            # Fallback to previous behavior if no color was assigned
            if not rgb:
                rand = 2
            else:
                rand = random.randint(0, len(colors) - 1)
            selected_color = colors[rand]
            
        if last_block is not None:
            for index in range(0, len(last_block[2])):
                self.draw_rect_on_coordinates(last_block[0] + last_block[2][index][0], 
                                             last_block[1] + last_block[2][index][1], "#FFA726")
        
        # Update global last_block
        import sys
        this_module = sys.modules['config']
        this_module.last_block = [x, y, coordinates]

        # Animate the placement of blocks
        self._animate_placement(x, y, coordinates, selected_color)
        
        # Update the game state
        for index in range(0, len(coordinates)):
            self.game.set_field(x + coordinates[index][0], y + coordinates[index][1], 1)
    
    def _animate_placement(self, x, y, coordinates, color):
        """Create a simple 'pop' animation for placing blocks"""
        import time
        
        # Track cells we're animating to clean them up between frames
        animated_items = []
        
        # First draw all blocks with a smaller size (scaling effect)
        for i in range(3):
            scale = 0.5 + (i * 0.25)  # Start at 50%, grow to 100%
            
            # Clean up previous animation items
            for item in animated_items:
                self.canvas.delete(item)
            animated_items = []
            
            for index in range(len(coordinates)):
                cx = x + coordinates[index][0]
                cy = y + coordinates[index][1]
                
                # Clear existing content in this cell
                items = self.canvas.find_overlapping(cx*50+1, cy*50+1, cx*50+49, cy*50+49)
                for item in items:
                    if self.canvas.type(item) == "rectangle":
                        self.canvas.delete(item)
                
                # Calculate scaled rectangle dimensions
                padding = int(25 * (1 - scale))
                scaled_x = cx * 50 + padding
                scaled_y = cy * 50 + padding
                scaled_width = 50 - (padding * 2)
                scaled_height = 50 - (padding * 2)
                
                # Create the background rectangle (lightgray)
                bg_rect = self.canvas.create_rectangle(cx*50, cy*50, cx*50+50, cy*50+50, 
                                                     fill="lightgray", outline="")
                
                # Draw the scaled rectangle
                rect = self.canvas.create_rectangle(scaled_x, scaled_y, 
                                                  scaled_x + scaled_width, scaled_y + scaled_height, 
                                                  fill=color, outline="")
                animated_items.extend([bg_rect, rect])
            
            # Update the display
            self.window.update()
            time.sleep(0.02 / 3)  # Keep animation fast
        
        # Clean up final animation items
        for item in animated_items:
            self.canvas.delete(item)
        
        # Finally draw the rectangles at full size with proper styling
        for index in range(len(coordinates)):
            self.draw_rect_on_coordinates(x + coordinates[index][0], y + coordinates[index][1], color)

    def remove_last_values(self, event):
        self.last_x = None
        self.last_y = None
        for index in range(0, len(self.last_preview)):
            lx = self.last_preview[index][0]
            ly = self.last_preview[index][1]
            if self.game.field[ly][lx] == 0:
                self.draw_rect(self.last_preview[index][0], self.last_preview[index][1], "lightgray")

    def draw_rect_on_coordinates(self, x, y, color):
        self.draw_rect(x, y, color)

    def clear_rect_on_coordinates(self, x, y):
        self.draw_rect(x, y, "lightgray")

    def draw_rect(self, x, y, color):
        x_pos = x * 50
        y_pos = y * 50
        
        # Find and delete any existing rectangles at this position
        items = self.canvas.find_overlapping(x_pos+1, y_pos+1, x_pos+49, y_pos+49)
        for item in items:
            if self.canvas.type(item) == "rectangle":
                self.canvas.delete(item)
        
        # Create gradient effect for more visual appeal
        if color != "lightgray":
            # Create main rectangle with the selected color
            rect_id = self.canvas.create_rectangle(x_pos, y_pos, x_pos + 50, y_pos + 50, fill=color, outline="")
            
            # Add a highlight effect on top-left
            highlight = self.canvas.create_polygon(
                x_pos, y_pos, x_pos + 50, y_pos, x_pos, y_pos + 50, 
                fill="", width=0, smooth=True
            )
            self.canvas.itemconfig(highlight, stipple="gray25", fill="white")
            
            # Add a shadow effect on bottom-right
            shadow = self.canvas.create_polygon(
                x_pos + 50, y_pos, x_pos + 50, y_pos + 50, x_pos, y_pos + 50, 
                fill="", width=0, smooth=True
            )
            self.canvas.itemconfig(shadow, stipple="gray25", fill="black")
        else:
            # Just a plain rectangle for empty cells
            self.canvas.create_rectangle(x_pos, y_pos, x_pos + 50, y_pos + 50, fill=color, outline="")
        
        self.restore_grid(self.img_id)

    def render_current_blocks(self):
        # Enhanced color palette with more vibrant options
        colors = [
            "#FF5252",   # Bright red
            "#FF4081",   # Pink
            "#7C4DFF",   # Deep purple
            "#536DFE",   # Indigo
            "#03A9F4",   # Light blue
            "#00BCD4",   # Cyan
            "#009688",   # Teal
            "#4CAF50",   # Green
            "#8BC34A",   # Light green
            "#FFEB3B",   # Yellow
            "#FFC107",   # Amber
            "#FF9800"    # Orange
        ]
        import random
        from config import rgb
        
        # First ensure all existing block canvases are properly handled
        for block in self.game.current_blocks:
            if hasattr(block, 'canvas') and block.canvas:
                try:
                    block.canvas.place_forget()  # Remove from view
                    block.canvas.destroy()       # Destroy the old canvas
                except:
                    pass  # Handle case where canvas might already be destroyed
                block.canvas = None  # Clear the reference
        
        # Restore block canvas overlay
        self.block_canvas.delete("all")
        self.block_canvas.create_image(0, 0, image=self.bc_overlay, anchor="nw")
        
        # Recreate canvases for all blocks
        for index in range(0, len(self.game.current_blocks)):
            block = self.game.current_blocks[index]
            
            # Ensure block has fresh canvas
            c = block.get_block_canvas()  # This will create a new canvas
            
            # Apply color to the block canvas
            if not hasattr(block, 'color'):
                if not rgb:
                    block.color = colors[2]  # Default color
                else:
                    block.color = colors[random.randint(0, len(colors) - 1)]
            
            # Apply the color to the block's rectangles
            try:
                for item in c.find_all():
                    if c.type(item) == "rectangle":
                        c.itemconfig(item, fill=block.color)
                
                c.place(x=50 + 166 * (index + 1) - 83 - int(c["width"]) / 2, 
                        y=75 + 500 + 25 + (62 - int(c["height"]) / 2))
            except:
                # If we encounter an error with this canvas, recreate it
                if hasattr(block, 'canvas') and block.canvas:
                    try:
                        block.canvas.destroy()
                    except:
                        pass
                    block.canvas = None
                
                # Create a fresh canvas and try again
                c = block.__create_block_canvas()
                block.canvas = c
                
                # Apply color
                for item in c.find_all():
                    if c.type(item) == "rectangle":
                        c.itemconfig(item, fill=block.color)
                
                c.place(x=50 + 166 * (index + 1) - 83 - int(c["width"]) / 2, 
                        y=75 + 500 + 25 + (62 - int(c["height"]) / 2))

    def restore_grid(self, img_id):
        # Delete the previous grid
        if img_id:
            self.canvas.delete(img_id)
        
        # Create a new grid that doesn't obscure the blocks
        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")


class GUILoseScreen:
    def __init__(self, window, game, lose_img):
        self.window = window
        self.game = game
        self.main = game.gui  # Store reference to the Main instance
        
        # Create the canvas for the lose screen
        self.canvas = Canvas(window, width=600, height=725, bg="#474747", highlightthickness=0)
        self.canvas.create_image(0, 0, image=lose_img, anchor="nw")
        
        # Display the score
        score_text = f"Your Score: {game.points}"
        self.canvas.create_text(300, 320, text=score_text, 
                                fill="white", font=("Segoe UI Light", 36))
        
        # Add a replay button
        replay_button = Button(self.canvas, text="Play Again", font=("Segoe UI", 16),
                              bg="#4CAF50", fg="white", padx=20, pady=10,
                              command=self.replay_game)
        replay_button_window = self.canvas.create_window(300, 400, window=replay_button)
        
        self.canvas.place(x=0, y=0)
    
    def replay_game(self):
        # Remove the lose screen
        self.canvas.destroy()
        
        # Use the current AI type when restarting the game
        ai_type = self.game.gui.ai_type
        
        # Clean up existing blocks before creating new ones
        for block in self.game.current_blocks:
            if hasattr(block, 'canvas') and block.canvas:
                try:
                    block.canvas.destroy()
                except:
                    pass  # Handle case where canvas might already be destroyed
                block.canvas = None  # Clear the reference
            
            # Remove is_used flag from blocks
            if hasattr(block, 'is_used'):
                delattr(block, 'is_used')
        
        # Reset the game field
        for i in range(len(self.game.field)):
            for j in range(len(self.game.field[i])):
                self.game.field[i][j] = 0
        
        # Reset points
        self.game.points = 0
        self.game.gui.points_label["text"] = "0"
        
        # Redraw the entire game board
        for y in range(10):
            for x in range(10):
                # First clear the rectangle in the data structure
                self.game.field[y][x] = 0
                # Then clear it visually using the correct references
                self.game.gui.canvas.create_rectangle(
                    x * 50, y * 50, 
                    x * 50 + 50, y * 50 + 50, 
                    fill="lightgray", outline=""
                )
        
        # Restore the grid overlay
        self.game.gui.restore_grid(self.game.gui.img_id)
        
        # Reset any global variables if needed
        import sys
        if 'config' in sys.modules:
            if hasattr(sys.modules['config'], 'last_block'):
                sys.modules['config'].last_block = None
        
        # Clear the block canvas before adding new blocks
        for item in self.game.gui.block_canvas.find_all():
            if self.game.gui.block_canvas.type(item) != "image":  # Preserve the background image
                self.game.gui.block_canvas.delete(item)
        
        # Generate new blocks
        self.game.current_blocks = []
        self.game.generate_blocks()
        self.game.gui.render_current_blocks()
