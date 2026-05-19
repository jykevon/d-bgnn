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
`adjMat`: the adjacency matrix that shows connections between lines if they share a box
`nodeFeatMat`: the node feature matrix, which shows the game state. the first column indicates if that line has been drawn and the second column indicates if the line is on the perimeter of the grid.

**Possible moves**

This can be easily calculated from the set of total edges minus the set of taken edges

**Random Model**

Returns a random move from the possible moves

**Heuristic Model**

While all the boxes in the game have a valence of greater than 1, pick a random move that doesn't make a box with a valence of one. Otherwise, pick a random move.

**Graph Neural Network**

The custom 2-layer spectral convolutional graph neural network has 64 hidden usits and uses ReLU as the activation function. For the message passing function, it is calculated as such: H = σ(A x N x W) where A is the adjacency matrix, N is the node feature matrix, and W is the weights supplied by `Keras.Dense`. The GNN has 64 hidden units, which is then condensed into 1 unit, which represents the q-value for a certain action. I chose to use the Adam optimizer with loss set to mean squared error. When building the model, you must use `None` as the first element of the shape node feature matrix input and the number of features for the second element, and both `None` for the shape adjacency matrix input so that it can play boards of different sizes.

**Training the Deep Q-Learning Model**

I implemented the Deep Q-Learning Algorithm from [Hugging Face](https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm). The main premise is to use normal Q-Learning but use a GNN to approximate the q-values. I used a deque from the collections library from Python. Each episode plays a singular game, where the action-value model utilizes an epsilon greedy strategy to play moves against the same action-value model without the epsilon greedy strategy and simulates the opponents reaction after what our model chose and generates a reward value. The reward calculation is +1 for every box our model maxes and -1 for every box the opponent model makes and illegal moves have a value of -1e9 to prevent the model making illegal moves. Then this is used to find the q-values for each of the interactions in the replay memory sample. The replay memory is then later sampled to train another model on creating a good target for q-values. Every 10 episodes the target action-value model's weights is set to the action-value model's weights. At the very end, the action-value model is saved.

**Playing Games**

I allowed input from the size of the board and number of games the user wants to play. For every game, reset the relevant data structures and create data structures to store the data you want to collect. Load the GNN you want to use and implement whatever models you would like the model to play against and allow each player to choose moves, which updates the relevent data stuctures. If you would like, print a string representation of the board after each move. If a model ever completes a box, allow them to go again. Once the board has been filled, find which model won by counting the number of boxes each player has then print the statistics from the game. At the end, you should have a sum of stats from all the games combined. Here is the string representation I chose to use where `x` is player 1 and `o` is player 2:
```
.---.   .   .
| o |   |    
.---.---.---.
| o | x |    
.---.---.---.
| x |   |    
.---.   .---.
```
