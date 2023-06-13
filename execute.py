"""
* File: project2.py
*
* Authors: Carter Steckbeck and Hyla Mosher
*
* Description: Implementation of multiple different AI agents, which includes Minimax,
*              Alpha-Beta Pruning, and Monte-Carlo Tree Search for the game Othello.
*              Also includes the different heuristics used for these searches.
*
"""

from othello import *
import random, sys
import math
import numpy as np
import matplotlib.pyplot as plt

class Node():
    def __init__(self, state, move, num_simulations, num_wins, children, parent):
        self.state = state # environment of the node
        self.move = move
        self.num_simulations = num_simulations
        self.num_wins = num_wins
        self.children = children
        self.parent = parent

class MoveNotAvailableError(Exception):
    """Raised when a move isn't available."""
    pass

class OthelloTimeOut(Exception):
    """Raised when a player times out."""
    pass

class OthelloPlayer():
    """Parent class for Othello players."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass

class RandomPlayer(OthelloPlayer):
    """Plays a random move."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return random.choice(state.available_moves())

class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        print("----- {}'s turn -----".format(state.current))
        print("Remaining time: {:0.2f}".format(remaining_time))
        print("Available moves are: ", available)
        move_string = input("Enter your move as 'r c': ")

        # Takes care of errant inputs and bad moves
        try:
            moveR, moveC = move_string.split(" ")
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)

def coin_eval(state):
    """ Heuristic that evaluates the number of coins for the player on the board """

    return state.count(state.current) - state.count(opposite_color(state.current))

def mobility_eval(state):
    """ Heuristic that evaluates the number of moves available for the player on the board """

    return len(state.available_moves())

def state_eval(state):
    """ Heuristic that evaluates the occupied squares on the board """

    evaluation = 0
    for r in range(len(state.board)):
        for c in range(len(state.board[0])):
            entry = (r,c)

            # Corners are given the highest weight
            if entry == (0,0) or entry == (0,7) or entry == (7,7) or entry == (7,0):
                if state.board[r][c] == state.current:
                    evaluation += 100

            # Squares directly diagnonal to the corner are given the worst weight
            if entry == (1,1) or entry == (6,6) or entry == (0,6) or entry == (6,0):
                if state.board[r][c] == state.current:
                    evaluation -= 50

            # Squares directly under, above, or next to the corner are given a bad weight
            if entry == (0,1) or entry == (1,0) or entry == (0,6) or entry == (1,7) or  \
                entry == (6, 0) or entry == (7,1) or entry == (7,6) or entry == (6,7):
                if state.board[r][c] == state.current:
                    evaluation -= 30

            # Squares a distance of two away from corner are given positive weight
            if entry == (0,2) or entry == (0,5) or entry == (7,5) or entry == (7,2) or \
                entry == (5,0) or entry == (5,7) or entry == (2,0) or entry == (2,7):
                if state.board[r][c] == state.current:
                    evaluation += 30

            # General edges are given positive weight since they cannot be flipped easily
            if entry == (3,0) or entry == (4,0) or entry == (0, 3) or entry == (0 ,4) or \
                entry == (7,3) or entry == (7,4) or entry == (3,7) or entry == (4, 7):
                if state.board[r][c] == state.current:
                    evaluation += 20

    return evaluation

def heuristic(state):
    square_eval = state_eval(state)
    piece_eval = coin_eval(state)
    mob_eval = mobility_eval(state)
    return square_eval + piece_eval + mob_eval

class MinimaxPlayer(OthelloPlayer):

    """ Player that uses the Minimax to play Othello """

    def minimaxAlg(self, child, depth, max_turn):
        """ Recursive algorithm that returns a leaf's heuristic using Minimax """

        # If at leaf or no more moves, return heuristic of state
        if depth == 0 or child.state.available_moves() == []:
            return (heuristic(child.state), None)

        # If it is MAX's turn, maximize value
        if max_turn:
            value = -math.inf
            state = child.state
            player = state.current
            best_move = None

            # Check all available moves
            for move in state.available_moves():
                new_state = state.apply_move(move)
                child = Node(new_state, move, 0, 0, [], None)

                # Call minimaxAlg recursively, set to False so it is now MIN's turn
                val, _= self.minimaxAlg(child, depth-1,  False)

                # If the value from leaf is greater than stored value, change
                # value and best_move
                if val > value:
                    value = val
                    best_move = move
            return value, best_move

        # If it is MIN's turn, minimizse
        else:
            value = math.inf
            best_move = None
            state = child.state

            # Check all available moves
            for move in state.available_moves():
                new_state = state.apply_move(move)
                child = Node(new_state, move, 0, 0, [], None)

                # Call minimaxAlg recursively, set to False so it is now MAX's turn
                val, _ = self.minimaxAlg(child, depth-1,  True)

                # If the value from leaf is less than stored value, change
                # value and best_move
                if val < value:
                    value = val
                    best_move = move
            return value, best_move

    def make_move(self, state, remaining_time):
        """ Plays move returned by Minimax algorithm """

        _, move = self.minimaxAlg(Node(state, None, 0, 0, [], None), 4, True)
        return move

class AlphaBetaPlayer(OthelloPlayer):

    """ Player that uses the Alpha-Beta Pruning to play Othello """

    def alphabetaAlg(self, child, alpha, beta, depth, max_turn):
        """ Recursive algorithm that returns a leaf's heuristic using Alpha-Beta Pruning """

         # If at leaf or no more moves, return heuristic of state
        if depth == 0 or child.state.available_moves() == []:
            return (heuristic(child.state), None)

        # If at MAX's turn, the maximize value
        if max_turn:
            state = child.state
            best_move = None

            # Check all available moves
            for move in state.available_moves():
                new_state = state.apply_move(move)
                child = Node(new_state, move, 0, 0, [], None)
                alp, _= self.alphabetaAlg(child, alpha, beta, depth-1, False)

                # If calculated alpha is greater than stored alpha, change stored alpha
                # and best_move
                if alp > alpha:
                    alpha = alp
                    best_move = move

                # If beta less than or equal to alpha, do not need to check other available moves
                if beta <= alpha:
                    break

            return alpha, best_move

        # If at MIN's turn, then minimize value
        else:
            best_move = None
            state = child.state
            for move in state.available_moves():
                new_state = state.apply_move(move)
                child = Node(new_state, move, 0, 0, [], None)
                bet, _ = self.alphabetaAlg(child, alpha, beta, depth-1, True)

                # If calculated beta is less than stored beta, change stored beta
                # and best_move
                if bet < beta:
                    beta = bet
                    best_move = move

                # If beta less than or equal to alpha, do not need to check other available moves
                if beta <= alpha:
                    break
            return beta, best_move

    def make_move(self, state, remaining_time):
        """ Plays move returned by Alpha-Beta algorithm """

        _, move = self.alphabetaAlg(Node(state, None, 0, 0, [], None), -math.inf, math.inf, 4, True)
        return move

class MonteCarloPlayer(OthelloPlayer):

    """ Player that uses Monte-Carlo Tree Search to calculate the best move"""

    def monteCarloTreeSearch(self, state):
        """ Implements Monte-Carlo Tree Search """

        # Create root, set player and expand the root
        root = Node(state, None, 0, 0, [], None)
        player = state.current
        root = self.expansion(root)

        while root.num_simulations < 400:

            # Select most promising node from the root
            curr_node = self.selection(root, player)

            # If the node is expanded, find most promising child
            # such that the child has not been expanded
            while curr_node.num_simulations > 1:
                curr_node = self.selection(curr_node, player)
            curr_node = self.expansion(curr_node)
            if curr_node.children == []:
                break

            # Rollout each child and update the tree
            for child in curr_node.children:
                val = self.rollout(child.state, player)
                root = self.backpropagate(val, child)

        maximum = -math.inf
        best_move = None

        # Find the best move out of the children
        for child in root.children:
            if child.num_simulations > maximum:
                maximum = child.num_simulations
                best_move = child.move
        return best_move

    def selection(self, node, player):
        """ Selects the best child node using UCB """

        selected_node = None
        maximum = -math.inf
        for child in node.children:
            if child.num_simulations == 0:
                return child
            selection = (child.num_wins / child.num_simulations) + \
                        (math.sqrt(2) * math.sqrt(np.log(node.num_simulations)/child.num_simulations))
            if selection > maximum:
                maximum = selection
                selected_node = child
        return selected_node

    def backpropagate(self, val, node):
        """ Updates the values in the tree after a rollout """

        node.num_simulations += 1
        node.num_wins += val
        while node.parent != None:
            parent = node.parent
            node.parent.num_wins += val
            node.parent.num_simulations += 1
            node = node.parent
        return node

    def expansion(self, node):
        """ Expands a node using all available moves in that state """

        children = []
        for move in node.state.available_moves():
            children.append(Node(node.state.apply_move(move), move, 0, 0, [], node))
        node.children = children
        return node

    def rollout(self, state, player):
        """ Rolls out a game with random moves until there is a winner """

        while state.available_moves() != []:
            state = state.apply_move(random.choice(state.available_moves()))
        if state.winner() == player:
            return 1
        if state.winner() == "draw":
            return 1/2
        return 0

    def make_move(self, state, remaining_time):
        """ Returns the best move calculated by Monte-Carlo Tree Search """

        return self.monteCarloTreeSearch(state)

class TournamentPlayer(OthelloPlayer):

    """ Modified Monte-Carlo Player """

    def monteCarloTreeSearch(self, state):
        root = Node(state, None, 0, 0, [], None)
        player = state.current
        root = self.expansion(root)

        # Alter depth based on move number
        if state.move_number < 10:
            iterations = 30
        else:
            iterations = 600
        while root.num_simulations < iterations:
            curr_node = self.selection(root, player)
            while curr_node.num_simulations > 1:
                curr_node = self.selection(curr_node, player)
            curr_node = self.expansion(curr_node)
            if curr_node.children == []:
                break
            for child in curr_node.children:
                val = self.rollout(child.state, player)
                root = self.backpropagate(val, child)

        maximum = -math.inf
        best_move = None
        for child in root.children:
            if child.num_simulations > maximum:
                maximum = child.num_simulations
                best_move = child.move
        return best_move

    def selection(self, node, player):
        """ Selects the best child node using UCB """

        selected_node = None
        maximum = -math.inf
        for child in node.children:
            if child.num_simulations == 0:
                return child
            selection = (child.num_wins / child.num_simulations) + \
                        (math.sqrt(2) * math.sqrt(np.log(node.num_simulations)/child.num_simulations))
            if selection > maximum:
                maximum = selection
                selected_node = child
        return selected_node

    def backpropagate(self, val, node):
        """ Updates the values in the tree after a rollout """

        node.num_simulations += 1
        node.num_wins += val
        while node.parent != None:
            parent = node.parent
            node.parent.num_wins += val
            node.parent.num_simulations += 1
            node = node.parent
        return node

    def expansion(self, node):
        """ Expands a node using all available moves in that state """

        children = []
        for move in node.state.available_moves():
            children.append(Node(node.state.apply_move(move), move, 0, 0, [], node))
        node.children = children
        return node

    def rollout(self, state, player):
        """ Rolls out a game with random moves until there is a winner """

        while state.available_moves() != []:
            state = state.apply_move(random.choice(state.available_moves()))
        if state.winner() == player:
            return 1
        if state.winner() == "draw":
            return 1/2
        return 0

    def make_move(self, state, remaining_time):
        """ Returns the best move calculated by Monte-Carlo Tree Search """

        return self.monteCarloTreeSearch(state)

# **********************************************************
# MONTE-CARLO WITH PURE ROLLOUT, NO EXPANSION OF CHILD NODES
# **********************************************************
# class TournamentPlayer1(OthelloPlayer):

#     def monteCarloTreeSearch(self, state):
#         player = state.current
#         root = Node(state, None, 0, 0, [], None)
#         for move in state.available_moves():
#             root.children.append(Node(state.apply_move(move), move, 0, 0, [], None))
#         if state.move_number < 10:
#             iterations = 30
#         else:
#             iterations = 600
#         for _ in range(iterations):
#             selected_node = self.selection(root)
#             temp_node = selected_node
#             root.num_simulations += 1
#             selected_node.num_simulations += 1
#             selected_node.num_wins += self.rollout(selected_node.state, player)
#             root.children[root.children.index(temp_node)] = selected_node

#         maximum = -math.inf
#         best_move = None
#         for child in root.children:
#             if child.num_simulations > maximum:
#                 maximum = child.num_simulations
#                 best_move = child.move
#         return best_move

#     def rollout(self, state, player):
#         while state.available_moves() != []:
#             state = state.apply_move(random.choice(state.available_moves()))
#         if state.winner() == player:
#             return 1
#         if state.winner() == "draw":
#             return 1/2
#         return 0

#     def selection(self, root):
#         selected_node = None
#         maximum = -math.inf
#         for child in root.children:
#             if child.num_simulations == 0:
#                 return child
#             selection = (child.num_wins / child.num_simulations) + \
#                         (math.sqrt(2) * math.sqrt(np.log(root.num_simulations)/child.num_simulations))
#             if selection > maximum:
#                 maximum = selection
#                 selected_node = child
#         return selected_node

#     def make_move(self, state, remaining_time):
#        return self.monteCarloTreeSearch(state)

# **********************************************************
# MONTE-CARLO TREE SEARCH WITH HEURISTIC
# **********************************************************

# class TournamentPlayer2(OthelloPlayer):
#     def coin_eval(self, state, player):
#         #Based on number of coins
#         if player == 'black':
#             max = 'black'
#             min = 'white'
#         else:
#             max = 'white'
#             min = 'black'
#         return (state.count(max) - state.count(min)/state.count(max) + state.count(min))

#     def mobility_eval(self, state):
#         # Limit the number of moves available to the opponent
#         return -len(state.available_moves())

#     def state_eval(self, state, player):
#         evaluation = 0
#         for r in range(len(state.board)):
#             for c in range(len(state.board[0])):
#                 entry = (r,c)
#                 if entry == (0,0) or entry == (0,7) or entry == (7,7) or entry == (7,0):
#                     if state.board[r][c] == opposite_color(player):
#                         evaluation -= 100
#                     if state.board[r][c] == player:
#                         evaluation += 100
#                 if entry == (1,1) or entry == (6,6) or entry == (0,6) or entry == (6,0):
#                     if state.board[r][c] == opposite_color(player):
#                         evaluation += 50
#                     if state.board[r][c] == player:
#                         evaluation -= 50
#                 if entry == (0,1) or entry == (1,0) or entry == (0,6) or entry == (1,7) or  \
#                    entry == (6, 0) or entry == (7,1) or entry == (7,6) or entry == (6,7):
#                     if state.board[r][c] == opposite_color(player):
#                         evaluation += 25
#                     if state.board[r][c] == player:
#                         evaluation -=25
#                 if entry == (0,2) or entry == (0,5) or entry == (7,5) or entry == (7,2) or \
#                    entry == (5,0) or entry == (5,7) or entry == (2,0) or entry == (2,7):
#                     if state.board[r][c] == opposite_color(player):
#                         evaluation -= 25
#                     if state.board[r][c] == player:
#                         evaluation += 25
#                 if entry == (3,0) or entry == (4,0) or entry == (0, 3) or entry == (0 ,4) or \
#                    entry == (7,3) or entry == (7,4) or entry == (3,7) or entry == (4, 7):
#                     if state.board[r][c] == opposite_color(player):
#                         evaluation -= 10
#                     if state.board[r][c] == player:
#                         evaluation += 10
#         return evaluation

#     def heuristic(self, state, player):
#         state_eval = self.state_eval(state, player)
#         coin_eval = self.coin_eval(state, player)
#         opponent_mobility_eval = self.mobility_eval(state)
#         return state_eval + (coin_eval * 25) + opponent_mobility_eval

#     def monteCarloTreeSearch(self, state):
#         children = []
#         player = state.current
#         root = Node(state, None, None, 0, 0)
#         for move in state.available_moves():
#             children.append(Node(state.apply_move(move), None, move, 0, 0))
#         for child in children:
#             winner = self.finish_game(state)
#             child.num_simulations += 1
#             if winner == player:
#                 child.num_wins += 1
#             if winner == "draw":
#                 child.num_wins += 1/2
#         root.num_simulations += len(children)
#         for _ in range(200):
#             maximum = -math.inf
#             selected_node = None
#             for child in children:
#                 selection = (child.num_wins / child.num_simulations) + \
#                             (math.sqrt(2) * math.sqrt(np.log(root.num_simulations)/child.num_simulations))
#                 if selection > maximum:
#                     maximum = selection
#                     selected_node = child
#             temp_node = selected_node
#             winner = self.finish_game(selected_node.state)
#             root.num_simulations += 1
#             selected_node.num_simulations += 1
#             if winner == player:
#                 selected_node.num_wins += 1
#             if winner == "draw":
#                 selected_node.num_wins += 1/2
#             children[children.index(temp_node)] = selected_node

#         maximum = -math.inf
#         best_move = None
#         for child in children:
#             print("Move: {}".format(child.move))
#             print("Num of simulations: {}".format(child.num_simulations))
#             print("Num of wins: {}".format(child.num_wins))
#             if child.num_simulations > maximum:
#                 maximum = child.num_simulations
#                 best_move = child.move
#         return best_move

#     def finish_game(self, state):
#         while state.available_moves() != []:
#             state = state.apply_move(random.choice(state.available_moves()))

#         return state.winner()
#         # while state.available_moves() != []:
#         #     if state.current == player:
#         #         max = -math.inf
#         #         player = state.current
#         #         best_move = None
#         #         for move in state.available_moves():
#         #             new_state = state.apply_move(move)
#         #             val = self.heuristic(new_state, player)
#         #             if val > max:
#         #                 max = val
#         #                 best_move = move
#         #         state = state.apply_move(best_move)
#         #     else:
#         #         state = state.apply_move(random.choice(state.available_moves()))

#         # return state.winner()


#     def make_move(self, state, remaining_time):
#        move = self.monteCarloTreeSearch(state)
#        return move

# **********************************************************
# MINIMAX PLAYER WITH DEPTH ALTERED DEPENDING ON MOVE NUMBER
# **********************************************************

# class TournamentPlayer3(OthelloPlayer): # Minimax with depth altered
#     def coin_eval(self, state):
#         #Based on number of coins
#         if state.current == 'black':
#             max = 'black'
#             min = 'white'
#         else:
#             max = 'white'
#             min = 'black'
#         return (state.count(max) - state.count(min)/state.count(max) + state.count(min))

#     def mobility_eval(self, state):
#         # Limit the number of moves available to the opponent
#         return -len(state.available_moves())

#     def state_eval(self, state):
#         evaluation = 0
#         for r in range(len(state.board)):
#             for c in range(len(state.board[0])):
#                 entry = (r,c)
#                 if entry == (0,0) or entry == (0,7) or entry == (7,7) or entry == (7,0):
#                     if state.board[r][c] == opposite_color(state.current):
#                         evaluation -= 100
#                     if state.board[r][c] == state.current:
#                         evaluation += 100
#                 if entry == (1,1) or entry == (6,6) or entry == (0,6) or entry == (6,0):
#                     if state.board[r][c] == opposite_color(state.current):
#                         evaluation += 50
#                     if state.board[r][c] == state.current:
#                         evaluation -= 50
#                 if entry == (0,1) or entry == (1,0) or entry == (0,6) or entry == (1,7) or  \
#                    entry == (6, 0) or entry == (7,1) or entry == (7,6) or entry == (6,7):
#                     if state.board[r][c] == opposite_color(state.current):
#                         evaluation += 25
#                     if state.board[r][c] == state.current:
#                         evaluation -= 25
#                 if entry == (0,2) or entry == (0,5) or entry == (7,5) or entry == (7,2) or \
#                    entry == (5,0) or entry == (5,7) or entry == (2,0) or entry == (2,7):
#                     if state.board[r][c] == opposite_color(state.current):
#                         evaluation -= 25
#                     if state.board[r][c] == state.current:
#                         evaluation += 25
#                 if entry == (3,0) or entry == (4,0) or entry == (0, 3) or entry == (0 ,4) or \
#                    entry == (7,3) or entry == (7,4) or entry == (3,7) or entry == (4, 7):
#                     if state.board[r][c] == opposite_color(state.current):
#                         evaluation -= 10
#                     if state.board[r][c] == state.current:
#                         evaluation += 10
#         return evaluation

#     def heuristic(self, state):
#         state_eval = self.state_eval(state)
#         coin_eval = self.coin_eval(state)
#         opponent_mobility_eval = self.mobility_eval(state)
#         return state_eval + coin_eval * 10 + opponent_mobility_eval

#     def minimaxAlg(self, child, depth, max_turn):
#         if depth == 0 or child.state.available_moves() == []:
#             return (child.utility, None)
#         if max_turn:
#             value = -math.inf
#             state = child.state
#             player = state.current
#             best_move = None
#             for move in state.available_moves():
#                 new_state = state.apply_move(move)
#                 child = Node(new_state, self.heuristic(new_state), move, 0, 0)
#                 val, _= self.minimaxAlg(child, depth-1,  False)
#                 if val > value:
#                     value = val
#                     best_move = move
#             return value, best_move
#         else:
#             value = math.inf
#             best_move = None
#             state = child.state
#             for move in state.available_moves():
#                 new_state = state.apply_move(move)
#                 child = Node(new_state, self.heuristic(new_state), move, 0, 0)
#                 val, _ = self.minimaxAlg(child, depth-1,  True)
#                 if val < value:
#                     value = val
#                     best_move = move
#             return value, best_move

#     def make_move(self, state, remaining_time):
#         if state.move_number < 10 or state.move_number > 60:
#             _, move = self.minimaxAlg(Node(new_state, None, 0, 0, [], None), 3, True)
#         else:
#             _, move = self.minimaxAlg(Node(new_state, None, 0, 0, [], None), 4, True)
#         return move

def main():
    """Plays the game."""

    black_wins = 0
    white_wins = 0
    for i in range(10):
        black_player = MonteCarloPlayer("black")
        white_player = TournamentPlayer("white")
        game = OthelloGame(black_player, white_player, verbose=True)
        winner = game.play_game()
        if winner == "black":
            black_wins += 1
        if winner == "white":
            white_wins += 1
    print("Black won {} times".format(black_wins))
    print("White won {} times".format(white_wins))

    # USED FOR GRAPHS

    # searches = ["MCTS", "MCTS Pure Rollout"]
    # x_axis_scale = np.arange(len(searches))
    # results_1 = [6, 5]
    # results_2 = [5, 4]
    # plt.bar(x_axis_scale - 0.2, results_1, 0.4, label = 'Black Player')
    # plt.bar(x_axis_scale + 0.2, results_2, 0.4, label = 'White Player')
    # plt.xticks(x_axis_scale, searches)
    # plt.xlabel("Type of Agent")
    # plt.ylabel("Number of Wins")
    # plt.title("Number of Wins for MCTS and MCTS Pure Rollout")
    # plt.legend(loc=0, prop={'size': 6})
    # plt.show()

if __name__ == "__main__":
    main()
