# sms-classifier
The project titled "SMS Classifier using Machine Learning" involves building a machine learning model to classify SMS messages as either spam or not spam (ham). This project follows a typical machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), data analysis, and model building. Below, I'll describe the different steps involved in this project:

1. **Importing Libraries**:
   - The project starts by importing necessary Python libraries such as Streamlit for creating a user interface, pickle for loading pre-trained models, NLTK for text preprocessing, and scikit-learn for building and using machine learning models.

2. **Loading Pre-Trained Models**:
   - The project loads pre-trained models, including a TF-IDF vectorizer (`vectorizer.pkl`) and a Multinomial Naive Bayes classifier (`model.pkl`) using the `pickle` module. These models were previously trained and saved.

3. **Text Preprocessing**:
   - The `transform_text` function is defined to preprocess the input SMS message. The preprocessing steps include:
     - Converting text to lowercase.
     - Tokenizing the text into words.
     - Removing non-alphanumeric characters.
     - Removing stopwords (common words like "the," "and," "is") using NLTK's stopwords list.
     - Stemming words using the Porter Stemmer to reduce them to their root forms.

4. **Streamlit User Interface**:
   - The Streamlit UI is set up with a title and a text input box for users to enter an SMS message they want to classify.

5. **Prediction**:
   - When the "Predict" button is clicked, the following steps are executed:
     - Check if the input SMS message is empty and display an error message if it is.
     - Preprocess the input SMS message using the `transform_text` function.
     - Vectorize the preprocessed message using the TF-IDF vectorizer.
     - Use the pre-trained Multinomial Naive Bayes model to make a prediction on whether the SMS is spam or not.
     - Display the prediction result as "Spam" or "Not Spam" to the user.

6. **Error Handling**:
   - The code includes error handling to catch and display any exceptions that may occur during prediction or model loading.

In addition to these steps, it's important to note that a typical machine learning project would involve some additional phases, such as:
- **Data Collection**: Gathering and obtaining the SMS data for training the model.
- **Data Cleaning**: Ensuring that the data is free of inconsistencies, missing values, and outliers.
- **Exploratory Data Analysis (EDA)**: Analyzing the dataset to gain insights, visualize patterns, and understand the characteristics of the data.
- **Feature Engineering**: Identifying and creating relevant features from the text data that can be used for model training.
- **Model Evaluation**: Assessing the model's performance using various evaluation metrics such as accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning**: Optimizing the model's hyperparameters to improve its performance.
- **Deployment**: If the model performs well, it can be deployed to production as an SMS spam classifier.

This project, as described, focuses primarily on the classification aspect of the machine learning pipeline, assuming that the data collection, cleaning, and EDA steps have already been completed.
