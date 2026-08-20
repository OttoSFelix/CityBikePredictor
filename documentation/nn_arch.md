# The acrhitecture of the neural network of the model

The model is a classification model that predicts how much each bike station will lose or gain bikes in the next 30 minutes. 
The classes are as follows:
- 0: the station has lost more than 3 bikes (high outflow)
- 1: the station has gained more than 3 bikes (high inflow)
- 2: the station hasn't lost or gained more than 3 bikes (stable)

The model has 23 input features:
- sin(time)
- cos(time)
- sin(weekday)
- cos(weekday)
- traffic volume
- bike amount
- rainfall
- station id (splitted into 16 embeddings)

### Time ⏱️
Time is used so that the model learns when the bikes have the most usage during the day. This is one of the most important features. Time is taken as the total seconds from 00:00 to current time and splitted into sine and cosine waves of the total seconds. This way the model correctly learns that time goes around the clock and starts again at midnight.

### Day of the week 📆
Day of the week is used so that the model learns the usual bike usage patterns and amounts for every weekday. Day of the week is one of 7 numerical values (0-6). The day is then splitted into sine and cosine waves of the numerical value. This way the model correctly learns that week is a continuous cycle where, for example, weekends have less traffic than on weekdays.

### Traffic volume 🚗
For each bike station, the mean traffic volume of the three nearest traffic sensors is used so that the model gets confirmation on the actual traffic at each moment and doesn't have to blindly guess the traffic solely from the time and day. The traffic volume is scaled using the *StandardScaler* from *sklearn.preprocessing*.

### Bike amount 🚲
Bike amount is used so that the model correclty learns the probabilities for high outflow and high inflow based on the current bike amount. For example, if a bike station has only 2 bikes, it is more likely to gain more bikes in the next 30 minutes rather than losing them. Additionally the bike station physically cannot lose more than 3 bikes if it only has 2 bikes to start. The bike amount is scaled using the *StandardScaler* from *sklearn.preprocessing*.

### Rainfall 💧
Rainfall is used so that the model learns that the heavier the rain, the less popular bikes are. Rainfall is given as mm/h for the last 10 minutes. Rainfall is scaled using the *StandardScaler* from *sklearn.preprocessing*.

### Station id 📍
Station id is used so that the model learns the possible popularities of each bike station. After all, not all bike station are located right at the heart of Helsinki or are the same size.

## Network size and form

![network](nn_form.png)

The form of the neural network is visualised above. The input layer has 23 neurons, the first hidden layers has 2048 neurons, the second hidden layer has 516 neurons, the third hidden layer has 128 neurons, the fourth hidden layer has 32 neurons and the network outputs a single value. The activation function for each hidden layer is *ReLU* and the other activation functions are linear (neutral). The first two hidden layers include a dropout of __ and each layer includes batch normalization before the activation function.