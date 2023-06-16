# Movie Review Sentiment Analysis Project using LSTM
This project is focused on sentiment analysis of movie reviews using Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN) architecture. The aim of the project is to develop a model that can accurately classify movie reviews as positive or negative based on their textual content, and deploying the model using Flask, a web framework for Python.

### Dataset
The project utilizes a dataset containing a collection of movie reviews labeled with their corresponding sentiment (positive or negative). The dataset is typically split into a training set and a test set. The training set is used to train the LSTM model, while the test set is used to evaluate its performance.

### LSTM Model
Long Short-Term Memory (LSTM) is a type of RNN that is effective in capturing long-term dependencies in sequential data. It has been widely used in natural language processing tasks, including sentiment analysis, due to its ability to process and remember information over long sequences.

The LSTM model for movie review sentiment analysis consists of the following components:

Embedding Layer: Converts each word in the input sequence into a dense vector representation, allowing the model to learn the contextual meaning of words.

LSTM Layer: Composed of LSTM units, this layer processes the sequential input data and captures the dependencies between words in the movie reviews.

Fully Connected Layer: Transforms the output of the LSTM layer into a prediction by applying a set of weights and biases. It maps the learned features to the sentiment labels (positive or negative).

Activation Function: Applied a sigmoid activation function to the output of the fully connected layer, producing the final sentiment prediction.

### Model Training and Evaluation
To train the LSTM model, the project follows these general steps:

Preprocess the movie review dataset by tokenizing the text, converting words to numerical representations, and performing any necessary data cleaning or normalization.

Split the dataset into a training set and a test set. The training set is used for model training, while the test set is used for evaluating the model's performance.

Configure the LSTM model architecture, including the number of LSTM units, embedding dimensions, and other hyperparameters.

Train the LSTM model on the training set, using techniques such as backpropagation and gradient descent to optimize the model's parameters.

### Deployment
Flask is a lightweight web framework for Python that allows developers to build web applications quickly and easily. It provides the tools and libraries needed to handle routing, request handling, and response rendering. Flask is known for its simplicity and flexibility, making it a popular choice for small to medium-sized web projects.
In the context of a sentiment analysis project, Flask can be used to deploy the trained LSTM model as a web application. This allows users to interact with the model through a user-friendly interface, enter movie reviews, and receive sentiment predictions in real time. Flask simplifies the process of handling user requests, processing input data, making predictions using the LSTM model, and rendering the results back to the user.

![Movie Analysis - Google Chrome 16-06-2023 22_12_47](https://github.com/Sameeruddin8/Movie-review-sentiment-analysis/assets/102674044/8fed2d10-2ef5-444a-94d6-46fd6f50dab0)


### Conclusion
This movie review sentiment analysis project demonstrates the application of LSTM neural networks for sentiment classification. By training an LSTM model on a movie review dataset and deploying it using Flask, users can easily obtain sentiment predictions for their own movie reviews. Feel free to explore and extend this project for further enhancements or other NLP tasks.
