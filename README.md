# Utilizing a Graph Neural Network (GNN) with Deep Q-Learning to Solve Dots and Boxes

**Abstract**

Many people are using popular algorithms such as Alpha Beta and Monte Carlo Tree Search to play common games such as Othello or Tic-Tac-Toe. As such, finding the best method to play these is games is always on the forefront of this field. In this paper, we are trying to play Dots and Boxes, a deterministic game that involves players drawing lines on a grid in order to complete boxes. However, the state space of Dots and Boxes increases exponentially, which makes algorithms such as Alpha Beta or Monte Carlo Tree Search have to do significantly more computation. Here, the main objective is to train a model to play the game with high proficiency and suffers less from the increase in board size.
To resolve this, we utilized a GNN in combination with Deep Q-Learning to try and reduce the time to find an optimal move. Through experimentation, the model achieved an win rate average of 81% across all experiments, and an average of 68.49% boxes captured, with the best performance against the heuristic model on a 5x5 board. It also achieved a superior time complexity, where increasing the board size had a negligible effect on the time to compute the best move. Additionally, the models performance is inversely proportional to the board size. Ultimately, this paper demonstrates that combining a GNN with Deep-Q Learning is a greatly viable solution for combinatorial games that can be represented as graphs.

To install required dependancies:
```pip install tensorflow keras numpy```

**How to use the program**

You can decided whether you want to train a new GNN, play games, or both by commenting or uncommenting their respective methods in the main method. Additionally, you can see a visual representation of the games by uncommenting `printBoard()` call in the `playGames()` method. To see the model the model plays, you can uncomment the print statement to the terminal printing the line number the model played. In order to avoid changing the current version of the GNN, you can rename the file for the model saved during training. Also, you can choose which model you use in the play moves method and the opponent model by uncommenting the certain model you can to play against and commenting the others.

**Data Structrures**
`BOARD`: a dictionary that maps an edge to every box it is connected to
`EDGES`: a set of the edges (0 to the number of edges - 1)
`TAKEN`: a list of the taken edges
`OWNERS`: a dictionary that maps the box index to the player that owns the box
`BOXES`: a dictionary that maps box index to the indicies of the lines that make the box

**Possible moves**
This can be easily calculated from the set of total edges minus the set of taken edges

**Random Model**
Returns a random move from the possible moves

**Heuristic Model**
While all the boxes in the game have a valence of greater than 1, pick a random move that doesn't make a box with a valence of one. Otherwise, pick a random move.


