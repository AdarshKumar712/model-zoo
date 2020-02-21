# Emojify using Embeddings
This is a tutorial to use Text Embeddings for text analysis.
Here, Embeddings layer along with LSTM has been used to perform sentimental analysis on short sentences and then corresponding emoji is predicted with an accuracy of 75% in the test data

## Dataset
Dataset has been taken from [Kaggle](https://www.kaggle.com/alvinrindra/emojify/download).
It contains three files:
  1. train_emoji.csv
  2. test_emoji.csv
  3. emojify_data.csv</li>
The model has been trained on the train.csv data and tested on test.csv data<br>
Dictionary matching for Emoji's:
      1. 0=>"💙",
      2. 1=> "🎾",
      3. 2=> "😄",
      4. 3=> "😞",
      5. 4=> "🍴"

## Key features of the model
<li> Pretrained Embeddings 'Glove' is used in Text Analysis as the dataset is small</li>
<li> Model is trained using LSTM layer along with dense layer on train.csv data</li><br>

## Reference
https://www.kaggle.com/alvinrindra/emojifier-with-rnn
