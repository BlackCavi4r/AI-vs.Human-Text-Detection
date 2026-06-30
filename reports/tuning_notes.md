# Model Tuning Notes

For the traditional machine learning models, I used `GridSearchCV` and selected the best setup with F1-score.

For the deep learning models, I used a smaller manual grid because FNN, LSTM, and CNN training takes longer on my laptop than the scikit-learn models. I tested practical values for sequence length, embedding size, hidden units or filters, dropout, learning rate, and epochs. The selected configuration is based on validation F1-score and then saved for the app.

This gives the project a repeatable tuning process without making the training setup heavier than it needs to be for the assignment.
