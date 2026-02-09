# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to develop a neural network regression model using a dataset created in Google Sheets with one numeric input and one numeric output. Regression is a supervised learning technique used to predict continuous values. A neural network is chosen because it can effectively learn both linear and non-linear relationships between input and output by adjusting its weights during training.

The model is trained using backpropagation to minimize a loss function such as Mean Squared Error (MSE). During each iteration, the training loss is calculated and updated. The training loss vs iteration plot is used to visualize the learning process of the model, where a decreasing loss indicates that the neural network is learning properly and converging toward an optimal solution.


## Neural Network Model

<img width="1115" height="695" alt="image" src="https://github.com/user-attachments/assets/14933a1a-6fdf-4649-b779-68a33dfca352" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:PRAGATHI KUMAR
### Register Number: 212224230200
```
#creating model class
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

#Function to train model
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information

<img width="274" height="523" alt="image" src="https://github.com/user-attachments/assets/93daba98-4d3a-4824-9dd0-91e3f46d9338" />


## OUTPUT
<img width="784" height="492" alt="image" src="https://github.com/user-attachments/assets/c2562d82-6731-4fba-a6c0-04240b1d5791" />


### Training Loss Vs Iteration Plot

<img width="720" height="572" alt="image" src="https://github.com/user-attachments/assets/f3540d89-0f61-4ded-8e75-ef24d89cd080" />


### New Sample Data Prediction

<img width="779" height="148" alt="image" src="https://github.com/user-attachments/assets/48810b9f-54d9-4b6b-a968-48faacc085cf" />


## RESULT
Thus the neural network regression model is developed using the given dataset.
