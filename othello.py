"""
CS 375
Support code for Othello board game.
"""

import copy, time, multiprocessing

def opposite_color(color):
    """Returns the other color"""
    assert color in ['black', 'white']
    return 'white' if color == 'black' else 'black'


class OthelloMove():
    """Represents a move in Othello with a (r, c) pair and the player who
    is playing that move."""

    def __init__(self, r, c, player):
        self.pair = (r, c)
        self.player = player

    def __str__(self):
        return "{} placing at {}".format(self.player, self.pair)

    def __repr__(self):
        return "({} {})".format(self.pair[0], self.pair[1])

    def __eq__(self, other):
        return self.pair == other.pair and self.player == other.player


class OthelloState():
    """Represents the state of an Othello game.
    The state includes the board (an 8x8 grid) and the current player."""

    def __init__(self):
        self.board = [['empty'] * 8 for _ in range(8)]
        self.current = 'black'
        self.move_number = 0
        self.board[3][3] = 'white'
        self.board[4][4] = 'white'
        self.board[3][4] = 'black'
        self.board[4][3] = 'black'

    def evaluation(self):
        """Difference between black and white pieces on board."""
        return self.count('black') - self.count('white')

    def game_over(self):
        """True if the game is over; false otherwise"""
        return self.available_moves() == []

    def winner(self):
        """ PRE:  self.game_over().  Return color of winner or 'draw' """
        assert self.game_over()
        ev = self.evaluation()
        return 'draw' if ev == 0 else 'black' if ev > 0 else 'white'

    def available_moves(self):
        """Returns a list of all available moves by current player for the
        current state."""

        # Contains (r, c) moves; not actual moves until turned into OthelloMove objects
        protomoves = []

        for r in range(8):
            for c in range(8):

                # Can only play in an empty square
                if self.board[r][c] == 'empty':

                    legal_move = False # We'll set to true to break out of loops when necessary

                    # We'll check in each direction for row and column
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr==0 and dc==0:
                                continue

                            if self.flanking(r + dr, c + dc, dr, dc,
                                             opposite_color(self.current),
                                             self.current):
                                protomoves.append((r, c))
                                legal_move = True
                                break

                        if legal_move:
                            break

        # Turn all protomoves into OthelloMoves
        return [OthelloMove(r, c, self.current) for r, c in protomoves]

    def flip(self, r, c, dr, dc, color):
        """ starting at r, c, moving in direction dr, dc, flip all of color found"""
        if r < 0 or c < 0 or r >= 8 or c >= 8 or self.board[r][c] != color:
            return
        self.board[r][c] = opposite_color(self.board[r][c])
        self.flip(r + dr, c + dc, dr, dc, color)

    def apply_move(self, move):
        """ move is an othello move that is applicable. Returns a new state. """

        new_state = copy.deepcopy(self)
        r,c = move.pair
        assert r >= 0 and c >= 0 and r < 8 and c < 8 and move.player == self.current
        assert self.board[r][c] == 'empty'
        directions = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr==0 and dc==0:
                    continue
                if self.flanking(r + dr, c + dc,
                                 dr, dc, opposite_color(self.current),
                                 self.current):
                    directions.append((dr, dc))

        for dr, dc in directions:
            new_state.flip(r + dr, c + dc, dr, dc, opposite_color(self.current))

        new_state.board[r][c] = move.player
        new_state.current = opposite_color(self.current)

        # If no legal moves, switch back to other player
        if new_state.available_moves() == []:
            new_state.current = move.player

        # Increment move_number in new_state
        new_state.move_number += 1

        return new_state

    def flank_help(self, r, c, dr, dc, row_color, end_color):
        """ True iff there is an unbroken sequence of row_color
        from (r,c) to a cell with end_color"""
        if r < 0 or r >= 8 or c < 0 or c >= 8 or self.board[r][c] == 'empty':
            return False
        if self.board[r][c] == end_color:
            return True
        return self.flank_help(r+dr, c+dc, dr, dc, row_color, end_color)

    def flanking(self, r, c, dr, dc, row_color, end_color):
        """True if (r, c) is row_color and there is an unbroken sequence of
        row_color from (r, c) to a cell with end_color"""
        return r >= 0 and c >= 0 and r < 8 and c < 8 and \
            self.board[r][c] == row_color and \
            self.flank_help(r + dr, c + dc, dr, dc, row_color, end_color)

    def count(self, color):
        """The number of pieces belonging to color on the board."""
        return sum([row.count(color) for row in self.board])

    def __str__(self):
        result =  "    0   1   2   3   4   5   6   7  \n"
        result += "  +---+---+---+---+---+---+---+---+\n"
        for r in range(8):
            result += "{} |".format(r)
            for c in range(8):
                col = self.board[r][c]
                result += '   |' if col == 'empty' else \
                          ' X |' if col == 'black' else \
                          ' O |'
            result += '\n'
            result += "  +---+---+---+---+---+---+---+---+\n"

        result += "============== STATUS ==============\n"
        result += "current player: {}\n".format(self.current)
        result += "black (X): {}\nwhite (O): {}".format(self.count('black'), self.count('white'))
        return result

################################################################################
# Below here for OthelloGame

def timed_make_move(player, board, remaining_time, return_list):
    """Used when timing moves to make sure player doesn't use more time than
    alotted. return_list contains the move that will be returned by reference"""
    move = player.make_move(board, remaining_time)
    return_list.append(move)

class OthelloGame:
    """Stores the game information as the game is played."""

    SECONDS_PER_PLAYER = 150.0

    def __init__(self, black, white, verbose=True):
        """Setup the game. If verbose=True, will print info about the game
        as it is played. Otherwise, play_game will simply return the winner."""

        self.black_player = black
        self.white_player = white
        self.verbose = verbose

        self.black_time = OthelloGame.SECONDS_PER_PLAYER
        self.white_time = OthelloGame.SECONDS_PER_PLAYER

        self.board = OthelloState()

        self.log(self.board)

    def log(self, *args):
        """Prints *args if verbose=True"""
        if self.verbose:
            print(*args)

    def timeout(self):
        """Called when a player runs out of time"""
        raise OthelloTimeOut()

    def play_game_timed(self):
        """Plays move until the game is over. Tracks time used by each player,
        and interupts game if someone runs out of time.
        NOTE: Doesn't work with HumanPlayer."""

        while not self.board.game_over():
            # Get the current player
            if self.board.current == 'black':
                player = self.black_player
                remaining_time = self.black_time
            else:
                player = self.white_player
                remaining_time = self.white_time

            # Used to retrieve the move by passing this list by reference
            manager = multiprocessing.Manager()
            move_list = manager.list()

            # The process that will let the player look for a move
            timer_process = multiprocessing.Process(target=timed_make_move,
                              args=(player, self.board, remaining_time, move_list))

            # Start getting a move
            start_time = time.time()
            timer_process.start()

            # Terminates if runs out of time
            timer_process.join(remaining_time)

            # If thread is still active, then the player ran out of time
            if timer_process.is_alive():
                # Terminate
                timer_process.terminate()
                timer_process.join()

                # End the game
                self.log("{} timed out!".format(self.board.current))
                self.log("Winner is", opposite_color(self.board.current))
                return opposite_color(self.board.current)

            # Calculate the time for the move
            end_time = time.time()
            move_time = end_time - start_time

            # Adjust the time
            if self.board.current == 'black':
                self.black_time -= move_time
            else:
                self.white_time -= move_time

            # Get the move out of the move_list and make the move
            move = move_list[0]
            self.board = self.board.apply_move(move)

            # Log the state of the game
            self.log("\n{}. {}".format(self.board.move_number, move))
            self.log(self.board)
            self.log("black time: {:0.2f}".format(self.black_time))
            self.log("white time: {:0.2f}".format(self.white_time))
            self.log("====================================")

        self.log("Winner is", self.board.winner())

        return self.board.winner()

    def play_game(self):
        """Plays move until the game is over. Tracks time used by each player,
        but does not interrupt game if someone runs out of time.
        NOTE: Works with HumanPlayer"""

        while not self.board.game_over():
            # Get the current player
            if self.board.current == 'black':
                player = self.black_player
                remaining_time = self.black_time
            else:
                player = self.white_player
                remaining_time = self.white_time

            # Start getting a move
            start_time = time.time()
            move = player.make_move(self.board, remaining_time)

            # Calculate the time for the move
            end_time = time.time()
            move_time = end_time - start_time

            # Adjust the time
            if self.board.current == 'black':
                self.black_time -= move_time
            else:
                self.white_time -= move_time

            # Make the move
            self.board = self.board.apply_move(move)

            # Log the state of the game
            self.log("\n{}. {}".format(self.board.move_number, move))
            self.log(self.board)
            self.log("black time: {:0.2f}".format(self.black_time))
            self.log("white time: {:0.2f}".format(self.white_time))
            self.log("====================================")

        self.log("Winner is", self.board.winner())

        return self.board.winner()
