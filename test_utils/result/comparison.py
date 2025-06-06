import pandas as pd
import os
os.chdir("C:/Users/vasu2/Desktop/UoA/Lecture (On class handson)/INFO 557/Project/graduate-project-kANNISTER/test_utils/result")

# Load friend's predictions CSV
friend_predictions = pd.read_csv("submission.csv")

my_predicted = pd.read_csv("submission1.csv")

# Compare predictions by matching your model's output with friend's output
correct_predictions = (my_predicted.iloc[:, 1:].values == friend_predictions.iloc[:, 1:].values)  # Exclude the 'text' column
wrong_pred = (my_predicted.iloc[:, 1:].values != friend_predictions.iloc[:, 1:].values)
print(wrong_pred)
accuracy = correct_predictions.mean()  # Calculate the accuracy as the mean of correct predictions
print(f"Your model's accuracy compared to friend's: {accuracy * 100:.2f}%")