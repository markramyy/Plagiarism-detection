from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import nltk  # type: ignore
import re
import tensorflow as tf  # type: ignore
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PlagiarismModelTrainer:
    def __init__(self, data_path='data/datap.csv', word2vec_path="plagiarism_model/word2vec_model.bin"):
        logger.debug("Initializing PlagiarismModelTrainer")
        self.data_path = data_path
        self.word2vec_path = word2vec_path
        self.model = None
        self.word2vec_model = None
        self.max_length = None
        self._ensure_nltk_downloads()

    def _ensure_nltk_downloads(self):
        """Download required NLTK resources"""
        logger.debug("Downloading NLTK resources")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def load_and_preprocess_data(self):
        """Load data from CSV and preprocess it"""
        logger.debug("Loading data from CSV")
        df = pd.read_csv(self.data_path)

        # Clean data
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        return df

    def clean_text(self, text):
        """Clean and tokenize text"""
        logger.debug("Cleaning and tokenizing text")
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return word_tokenize(text)

    def train_word2vec(self, df):
        """Train Word2Vec model on the dataset"""
        logger.debug("Training Word2Vec model")
        df['tokenized_source'] = df['source_text'].apply(self.clean_text)
        df['tokenized_plagiarized'] = df['plagiarized_text'].apply(self.clean_text)

        sentences = df['tokenized_source'].tolist() + df['tokenized_plagiarized'].tolist()
        self.word2vec_model = Word2Vec(sentences=sentences, vector_size=200, window=7, min_count=1, sg=1, workers=4)
        self.word2vec_model.save(self.word2vec_path)

        return self.word2vec_model

    def text_to_embedding(self, text, max_length):
        """Convert text to embeddings using Word2Vec"""
        logger.debug("Converting text to embeddings")
        words = word_tokenize(text.lower())
        embedding_matrix = np.zeros((max_length, self.word2vec_model.vector_size))

        for i, word in enumerate(words[:max_length]):
            if word in self.word2vec_model.wv:
                embedding_matrix[i] = self.word2vec_model.wv[word]

        return embedding_matrix

    def prepare_data_for_training(self, df):
        """Prepare data for model training"""
        logger.debug("Preparing data for training")
        sentence_lengths = df['source_text'].apply(lambda x: len(x.split()))
        self.max_length = int(np.percentile(sentence_lengths, 90))
        print(f"Max Sequence Length: {self.max_length}")

        X_source = np.array([self.text_to_embedding(text, self.max_length) for text in df['source_text']])
        X_plagiarized = np.array([self.text_to_embedding(text, self.max_length) for text in df['plagiarized_text']])
        X = np.concatenate((X_source, X_plagiarized), axis=1)
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        """Build the LSTM model architecture"""
        logger.debug("Building LSTM model")
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                          input_shape=input_shape),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
            BatchNormalization(),
            LSTM(64, dropout=0.3, recurrent_dropout=0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        logger.debug("Compiling model")
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )

        return model

    def train(self, epochs=10, batch_size=128):
        """Train the plagiarism detection model"""
        logger.debug("Training plagiarism detection model")
        df = self.load_and_preprocess_data()

        # Train word2vec if needed or load existing model
        try:
            self.word2vec_model = Word2Vec.load(self.word2vec_path)
            print("Loaded existing Word2Vec model")
        except Exception:
            print("Training new Word2Vec model")
            self.train_word2vec(df)

        X_train, X_test, y_train, y_test = self.prepare_data_for_training(df)

        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        logger.debug("Training model")
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[es]
        )

        # Evaluate the model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return self.model, self.word2vec_model, self.max_length

    def save_model(self, model_path="plagiarism_model/plagiarism_model.h5"):
        """Save the trained model"""
        logger.debug(f"Saving model to {model_path}")
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Train a model first.")


if __name__ == "__main__":
    trainer = PlagiarismModelTrainer()
    model, word2vec_model, max_length = trainer.train()
    trainer.save_model()
