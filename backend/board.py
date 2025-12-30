import random

class MinesweeperBoard:
    DEFAULT_NEIGHBORS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def __init__(
        self,
        width,
        height,
        num_mines,
        seed=None,
        custom_mask=None,
        reserve_first_click_area=True,
        reserve_radius=1
    ):
        """
        reserve_first_click_area:
            If True (default), mines are placed lazily on the first reveal(),
            excluding a (2*reserve_radius+1) x (2*reserve_radius+1) block
            centered on that first click. This guarantees:
                - The first clicked cell is not a mine
                - Its neighbors (in that block) are not mines
                - With the default mask, the first cell will be a 0 and
                  your flood-fill will open a small region.

        reserve_radius:
            Radius for the reserved area around the first click (default 1 → 3x3).
        """
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.seed = seed
        self.custom_mask = custom_mask if custom_mask is not None else self.DEFAULT_NEIGHBORS

        # Behavior flags for lazy placement / “nice opening”
        self.reserve_first_click_area = reserve_first_click_area
        self.reserve_radius = reserve_radius
        self.mines_placed = False

        self.board = []   # -1 = mine, 0–8 = adjacent mine counts
        self.revealed = []  # bool grid
        self.flags = []     # bool grid

        self._init_board()

    def _init_board(self):
        if self.seed is not None:
            random.seed(self.seed)

        # Start with an empty board; we may delay mine placement
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.revealed = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.flags = [[False for _ in range(self.width)] for _ in range(self.height)]

        if not self.reserve_first_click_area:
            # Original behavior: place mines immediately
            self._place_mines()
            self._compute_adjacent_counts()
            self.mines_placed = True
        else:
            # Lazy behavior: wait until the first reveal() to place mines
            self.mines_placed = False

    def _place_mines(self, exclude_cells=None):
        """
        Place mines randomly on the board, optionally excluding a set of cells.
        exclude_cells: iterable of (row, col) coordinates that must NOT contain a mine.
        """
        if exclude_cells is None:
            exclude_set = set()
        else:
            exclude_set = set(exclude_cells)

        all_coords = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in exclude_set
        ]

        if self.num_mines > len(all_coords):
            raise ValueError(
                f"Cannot place {self.num_mines} mines: only {len(all_coords)} "
                f"available cells after excluding reserved area."
            )

        # Clear any existing mines before placing (for safety)
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] == -1:
                    self.board[r][c] = 0

        mine_coords = random.sample(all_coords, self.num_mines)
        for r, c in mine_coords:
            self.board[r][c] = -1

    def _compute_adjacent_counts(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] == -1:
                    continue
                count = 0
                for dr, dc in self.custom_mask:
                    nr, nc = r + dr, c + dc
                    if self.is_valid_coord(nr, nc) and self.board[nr][nc] == -1:
                        count += 1
                self.board[r][c] = count

    def is_valid_coord(self, row, col):
        try:
            row = int(row)
            col = int(col)
        except (ValueError, TypeError):
            return False
        return 0 <= row < self.height and 0 <= col < self.width

    def _reserved_cells_around(self, row, col):
        """
        Compute the reserved opening area around (row, col).
        By default this is a 3x3 block (radius=1) centered on the first click,
        clipped to the board boundaries.
        """
        cells = []
        for dr in range(-self.reserve_radius, self.reserve_radius + 1):
            for dc in range(-self.reserve_radius, self.reserve_radius + 1):
                rr, cc = row + dr, col + dc
                if self.is_valid_coord(rr, cc):
                    cells.append((rr, cc))
        return cells

    def reveal(self, row: int, col: int) -> bool:
        """
        Reveal behavior:
        - On the very first reveal (when reserve_first_click_area is True),
          place mines excluding a reserved area around (row, col),
          then compute counts.
        - Normal reveal:
            * if unrevealed and not flagged:
                - hitting a mine → return True
                - 0 → flood fill using custom_mask
                - number → reveal only this cell
            * if already revealed and number > 0:
                - “mass reveal” neighbors when flag count matches the number.
        Returns True if *any* mine gets revealed by this action (including flood fill).
        """
        if not self.is_valid_coord(row, col):
            return False

        # Lazy mine placement with a guaranteed opening around the first click
        if self.reserve_first_click_area and not self.mines_placed:
            reserved = self._reserved_cells_around(row, col)
            self._place_mines(exclude_cells=reserved)
            self._compute_adjacent_counts()
            self.mines_placed = True

        # CASE 1: normal reveal (if not yet revealed and not flagged)
        if not self.revealed[row][col] and not self.flags[row][col]:
            self.revealed[row][col] = True

            # Check if the current cell itself is a mine
            if self.board[row][col] == -1:
                return True  # Current cell is a mine

            # If it's a '0', flood fill. Accumulate mine_hit status from recursive calls.
            if self.board[row][col] == 0:
                mine_hit_during_flood_fill = False
                # Use the custom_mask for 0-reveal flood fill
                # because the '0' was determined by the custom_mask.
                for dr, dc in self.custom_mask:
                    nr, nc = row + dr, col + dc
                    if self.is_valid_coord(nr, nc):
                        if self.reveal(nr, nc):
                            mine_hit_during_flood_fill = True
                return mine_hit_during_flood_fill

            # If it's a number (1–8), it's revealed, no mine hit by this specific action.
            return False

        # CASE 2: mass reveal (if already revealed and is a number > 0)
        if self.revealed[row][col] and self.board[row][col] > 0:
            flagged = 0
            to_reveal = []

            for dr, dc in self.custom_mask:
                nr, nc = row + dr, col + dc
                if self.is_valid_coord(nr, nc):
                    if self.flags[nr][nc]:
                        flagged += 1
                    elif not self.revealed[nr][nc]:
                        to_reveal.append((nr, nc))

            if flagged == self.board[row][col]:
                mine_hit_on_mass_reveal = False
                for nr, nc in to_reveal:
                    if self.reveal(nr, nc):
                        mine_hit_on_mass_reveal = True
                return mine_hit_on_mass_reveal

        # If action is invalid (revealing flagged cell, or already revealed cell
        # not leading to mass reveal)
        return False

    def flag(self, row, col):
        if self.is_valid_coord(row, col) and not self.revealed[row][col]:
            self.flags[row][col] = not self.flags[row][col]

    def is_mine(self, row, col):
        return self.is_valid_coord(row, col) and self.board[row][col] == -1

    def is_revealed(self, row, col):
        return self.is_valid_coord(row, col) and self.revealed[row][col]

    def is_flagged(self, row, col):
        return self.is_valid_coord(row, col) and self.flags[row][col]

    def is_complete(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] != -1 and not self.revealed[r][c]:
                    return False
        return True

    def get_visible_state(self, game_over_flag=False, game_won_flag=False):
        state = []
        for r in range(self.height):
            row_cells = []
            for c in range(self.width):
                is_mine_cell = (self.board[r][c] == -1)

                if game_over_flag:
                    if is_mine_cell:
                        if game_won_flag:
                            # Game won, all mines are effectively "flagged"
                            row_cells.append("F")
                        else:  # Game lost
                            if self.revealed[r][c] and self.board[r][c] == -1:
                                # This is the mine that was clicked
                                row_cells.append("*")  # Exploded mine
                            elif self.flags[r][c]:
                                # Correctly flagged mine
                                row_cells.append("F")
                            else:
                                # Other unflagged, unrevealed mines
                                row_cells.append("M")
                    else:  # Not a mine
                        if self.flags[r][c]:
                            # Incorrectly flagged non-mine
                            row_cells.append("X")
                        elif self.revealed[r][c]:
                            row_cells.append(self.board[r][c])
                        else:
                            # Unrevealed non-mine cell in a game over state (loss)
                            # Show as None (still hidden) or could show its number
                            row_cells.append(None)
                else:
                    if self.flags[r][c]:
                        row_cells.append("F")
                    elif not self.revealed[r][c]:
                        row_cells.append(None)
                    else:
                        row_cells.append(self.board[r][c])
            state.append(row_cells)
        return state

    def print_debug_board(self):
        for r in range(self.height):
            row = ""
            for c in range(self.width):
                if self.board[r][c] == -1:
                    row += " * "
                else:
                    row += f" {self.board[r][c]} "
            print(row)
