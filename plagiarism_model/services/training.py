from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import nltk  # type: ignore
import re
import tensorflow as tf  # type: ignore
import logging
import gc
import os
from tqdm import tqdm  # type: ignore
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PlagiarismModelTrainer:
    def __init__(self, data_path='data/dataset.csv', word2vec_path="plagiarism_model/word2vec_model.bin", batch_size=1000):
        logger.debug("Initializing PlagiarismModelTrainer")
        self.data_path = data_path
        self.word2vec_path = word2vec_path
        self.model = None
        self.word2vec_model = None
        self.max_length = None
        self.batch_size = batch_size
        self._ensure_nltk_downloads()

    def _ensure_nltk_downloads(self):
        """Download required NLTK resources"""
        logger.debug("Downloading NLTK resources")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def load_and_preprocess_data(self):
        """Load data from CSV and preprocess it with reduced memory usage"""
        logger.debug("Loading data from CSV")

        # Determine file size to see if we need chunking
        file_size = os.path.getsize(self.data_path) / (1024 * 1024)  # Size in MB

        if file_size > 200:  # If file is larger than 200MB, read in chunks
            logger.info(f"Large file detected ({file_size:.2f} MB). Using chunked reading.")

            chunks = []
            for chunk in pd.read_csv(self.data_path, chunksize=10000):
                # Clean within chunk
                chunk = chunk.dropna()
                chunk = chunk.drop_duplicates()
                chunks.append(chunk)

            df = pd.concat(chunks)
            # Final deduplication across all chunks
            df = df.drop_duplicates()

        else:
            # Regular loading for smaller files
            df = pd.read_csv(self.data_path)
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)

        # Sample the data if it's too large (adjust sample size as needed)
        if len(df) > 100000:
            logger.info(f"Dataset very large ({len(df)} rows). Sampling 50k rows for training.")
            df = df.sample(n=50000, random_state=42)

        return df

    def clean_text(self, text):
        """Clean and tokenize text with improved preprocessing"""
        if not isinstance(text, str):
            return []

        # Enhanced text preprocessing
        text = text.lower()
        # Remove punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s\.]', '', text)
        # Split into sentences first for better context
        sentences = text.split('.')
        tokens = []

        for sentence in sentences:
            if sentence.strip():
                # Tokenize each sentence
                sentence_tokens = word_tokenize(sentence.strip())
                tokens.extend(sentence_tokens)

        # Remove very common words that don't help with plagiarism detection
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been'}
        tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def calculate_ngram_features(self, source_tokens, plagiarized_tokens):
        """Calculate n-gram overlap features for improved pattern detection"""

        def get_ngrams(tokens, n):
            return set(zip(*[tokens[i:] for i in range(n)]))

        # Calculate bigram and trigram features
        source_bigrams = get_ngrams(source_tokens, 2) if len(source_tokens) > 1 else set()
        plag_bigrams = get_ngrams(plagiarized_tokens, 2) if len(plagiarized_tokens) > 1 else set()

        source_trigrams = get_ngrams(source_tokens, 3) if len(source_tokens) > 2 else set()
        plag_trigrams = get_ngrams(plagiarized_tokens, 3) if len(plagiarized_tokens) > 2 else set()

        # Calculate Jaccard similarity for bigrams and trigrams
        bigram_sim = len(source_bigrams.intersection(plag_bigrams)) / max(1, len(source_bigrams.union(plag_bigrams)))
        trigram_sim = len(source_trigrams.intersection(plag_trigrams)) / max(1, len(source_trigrams.union(plag_trigrams)))

        return bigram_sim, trigram_sim

    def train_word2vec_batched(self, df):
        """Train Word2Vec model on the dataset in batches"""
        logger.debug("Training Word2Vec model in batches")

        # Process in batches to avoid memory issues
        sentences = []

        # Process in tqdm batches for progress visibility
        for i in tqdm(range(0, len(df), self.batch_size), desc="Tokenizing text"):
            batch = df.iloc[i: i + self.batch_size]

            # Process source texts
            batch_source_tokens = [self.clean_text(text) for text in batch['source_text']]
            sentences.extend([tokens for tokens in batch_source_tokens if tokens])

            # Process plagiarized texts
            batch_plag_tokens = [self.clean_text(text) for text in batch['plagiarized_text']]
            sentences.extend([tokens for tokens in batch_plag_tokens if tokens])

            # Explicitly clear memory
            del batch_source_tokens, batch_plag_tokens
            gc.collect()

        # Train Word2Vec with optimized parameters
        logger.debug(f"Training Word2Vec on {len(sentences)} sentences")
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=100,  # Reduced from 200 to save memory
            window=5,
            min_count=2,
            sg=1,  # Skip-gram
            workers=4,
            batch_words=10000  # Process more words per batch
        )
        self.word2vec_model.save(self.word2vec_path)

        # Clear memory
        del sentences
        gc.collect()

        return self.word2vec_model

    def text_to_embedding(self, text, max_length):
        """Convert text to embeddings using Word2Vec with improved representation"""
        if not isinstance(text, str):
            text = ""

        # Preprocess text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text)

        # Create embedding matrix
        embedding_matrix = np.zeros((max_length, self.word2vec_model.vector_size))

        # Compute document-level features
        if words:
            # Get average of all word vectors for the document
            doc_vector = np.zeros(self.word2vec_model.vector_size)
            word_count_in_vocab = 0

            for word in words:
                if word in self.word2vec_model.wv:
                    doc_vector += self.word2vec_model.wv[word]
                    word_count_in_vocab += 1

            if word_count_in_vocab > 0:
                doc_vector /= word_count_in_vocab

        # Fill embedding matrix with individual word vectors
        for i, word in enumerate(words[:max_length]):
            if word in self.word2vec_model.wv:
                word_vector = self.word2vec_model.wv[word]

                # Combine word vector with position information
                position_factor = 1.0 - (i / max_length) * 0.2  # Position weighting
                embedding_matrix[i] = word_vector * position_factor

        return embedding_matrix

    def calculate_lcs_ratio(self, source_tokens, plagiarized_tokens):
        """Calculate longest common subsequence ratio for better sequence matching"""
        def lcs_length(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])

            return L[m][n]

        lcs = lcs_length(source_tokens, plagiarized_tokens)
        total_length = max(1, len(source_tokens) + len(plagiarized_tokens))
        return 2 * lcs / total_length  # Normalized ratio

    def prepare_data_for_training_with_similarity(self, df):
        """Prepare data with additional similarity features"""
        logger.debug("Preparing data with similarity features")

        # Calculate max sequence length as before
        if len(df) > 10000:
            sample_df = df.sample(n=10000, random_state=42)
            sentence_lengths = sample_df['source_text'].apply(lambda x: len(str(x).split()))
        else:
            sentence_lengths = df['source_text'].apply(lambda x: len(str(x).split()))

        self.max_length = min(int(np.percentile(sentence_lengths, 90)), 20)
        print(f"Max Sequence Length: {self.max_length}")

        # Process data in batches
        X_source_list = []
        X_plagiarized_list = []
        similarity_features_list = []
        y_list = []

        # Compute similarity features for each pair
        for i in tqdm(range(0, len(df), self.batch_size), desc="Creating embeddings and similarity features"):
            batch = df.iloc[i: i + self.batch_size]

            similarity_features = []

            for idx, row in batch.iterrows():
                source = str(row['source_text'])
                plagiarized = str(row['plagiarized_text'])

                # Clean and tokenize texts
                source_tokens = self.clean_text(source)
                plagiarized_tokens = self.clean_text(plagiarized)  # Fix: This line was missing proper tokenization
                source_tokens_set = set(source_tokens)
                plagiarized_tokens_set = set(plagiarized_tokens)

                if len(source_tokens_set) == 0 or len(plagiarized_tokens_set) == 0:
                    jaccard = 0
                    containment = 0
                    length_ratio = 0
                    bigram_sim = 0
                    trigram_sim = 0
                    lcs_ratio = 0
                else:
                    # Jaccard similarity
                    jaccard = len(source_tokens_set.intersection(plagiarized_tokens_set)) / len(source_tokens_set.union(plagiarized_tokens_set))

                    # Containment measure
                    containment = len(source_tokens_set.intersection(plagiarized_tokens_set)) / len(source_tokens_set)

                    # Length ratio
                    length_ratio = min(len(source), len(plagiarized)) / max(len(source), len(plagiarized)) if max(len(source), len(plagiarized)) > 0 else 0

                    # N-gram features
                    bigram_sim, trigram_sim = self.calculate_ngram_features(source_tokens, plagiarized_tokens)

                    # Add longest common subsequence ratio feature
                    lcs_ratio = self.calculate_lcs_ratio(source_tokens, plagiarized_tokens)

                # Add expanded similarity features
                similarity_features.append([jaccard, containment, length_ratio, bigram_sim, trigram_sim, lcs_ratio])

            # Convert to source and plagiarized embeddings as before
            X_source_batch = np.array([self.text_to_embedding(str(text), self.max_length) for text in batch['source_text']])
            X_plagiarized_batch = np.array([self.text_to_embedding(str(text), self.max_length) for text in batch['plagiarized_text']])

            # Store the batches
            X_source_list.append(X_source_batch)
            X_plagiarized_list.append(X_plagiarized_batch)
            similarity_features_list.append(np.array(similarity_features))
            y_list.append(batch['label'].values)

            # Clear memory
            del X_source_batch, X_plagiarized_batch, similarity_features, batch
            gc.collect()

        # Combine all batches
        X_source = np.vstack(X_source_list)
        X_plagiarized = np.vstack(X_plagiarized_list)
        similarity_features = np.vstack(similarity_features_list)
        y = np.concatenate(y_list)

        # Clear memory
        del X_source_list, X_plagiarized_list, similarity_features_list, y_list
        gc.collect()

        # Create combined model inputs
        X_text = np.concatenate((X_source, X_plagiarized), axis=1)

        # Split data
        X_text_train, X_text_test, X_sim_train, X_sim_test, y_train, y_test = train_test_split(
            X_text, similarity_features, y, test_size=0.2, random_state=42
        )

        # Reshape for LSTM
        X_text_train = X_text_train.reshape(X_text_train.shape[0], X_text_train.shape[1], X_text_train.shape[2])
        X_text_test = X_text_test.reshape(X_text_test.shape[0], X_text_test.shape[1], X_text_test.shape[2])

        return X_text_train, X_text_test, X_sim_train, X_sim_test, y_train, y_test

    def build_combined_model(self, text_input_shape, sim_input_shape):
        """Build a combined model with enhanced architecture and regularization"""
        # Text input branch
        text_input = tf.keras.Input(shape=(text_input_shape[1], text_input_shape[2]), name='text_input')

        # Process text with Bidirectional LSTM
        x_text = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        )(text_input)
        x_text = BatchNormalization()(x_text)
        x_text = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x_text)
        x_text = BatchNormalization()(x_text)

        # Add attention layer
        attention = tf.keras.layers.Dense(1, activation='tanh')(x_text)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)  # 128 = 64*2 for bidirectional
        attention = tf.keras.layers.Permute([2, 1])(attention)

        # Apply attention to LSTM outputs
        x_text = tf.keras.layers.multiply([x_text, attention])

        # Define Lambda layer with explicit output_shape
        def sum_over_axis(x):
            return tf.keras.backend.sum(x, axis=1)

        # Define output shape function for the Lambda layer
        def sum_output_shape(input_shape):
            return (input_shape[0], input_shape[2])

        # Use Lambda layer with explicit output_shape
        x_text = tf.keras.layers.Lambda(
            sum_over_axis,
            output_shape=sum_output_shape
        )(x_text)

        x_text = Dense(
            64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x_text)
        x_text = BatchNormalization()(x_text)

        # Similarity features branch - expanded for your new features
        sim_input = tf.keras.Input(shape=(sim_input_shape[1],), name='similarity_input')
        x_sim = Dense(32, activation='relu')(sim_input)
        x_sim = BatchNormalization()(x_sim)
        x_sim = Dense(16, activation='relu')(x_sim)
        x_sim = BatchNormalization()(x_sim)

        # Combine both branches
        combined = tf.keras.layers.concatenate([x_text, x_sim])

        # Output layers with improved regularization
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(combined)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)

        # Create model with multiple inputs
        model = tf.keras.Model(inputs=[text_input, sim_input], outputs=output)

        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()
            ]
        )

        return model

    def augment_training_data(self, df):
        """Augment training data to improve generalization"""
        logger.debug("Augmenting training data")

        augmented_samples = []
        # Only augment a subset to keep it manageable
        subset = df.sample(min(1000, len(df)), random_state=42)

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Augmenting data"):
            source_text = str(row['source_text'])
            plagiarized_text = str(row['plagiarized_text'])
            label = row['label']

            # For positive examples (label=1), create new positive examples with slight modifications
            if label == 1:
                # Create modified version by word substitution or reordering
                words = source_text.split()
                if len(words) > 5:
                    # Swap some adjacent words to create a variant
                    for i in range(min(3, len(words) - 1)):
                        idx = np.random.randint(0, len(words) - 1)
                        words[idx], words[idx + 1] = words[idx + 1], words[idx]

                    modified_source = " ".join(words)
                    augmented_samples.append({
                        'source_text': modified_source,
                        'plagiarized_text': plagiarized_text,
                        'label': 1  # Still a positive example
                    })
            else:  # For negative examples
                # Sometimes create more challenging negative examples
                if len(source_text.split()) > 5 and len(plagiarized_text.split()) > 5:
                    # Take part of source and part of plagiarized to create a "borderline" negative
                    s_words = source_text.split()
                    p_words = plagiarized_text.split()

                    s_half = s_words[:len(s_words) // 2]
                    p_half = p_words[len(p_words) // 2:]

                    modified_text = " ".join(s_half + p_half)

                    augmented_samples.append({
                        'source_text': source_text,
                        'plagiarized_text': modified_text,
                        'label': 0  # Still a negative example but more challenging
                    })

        # Create DataFrame from augmented samples and combine with original
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"Added {len(augmented_df)} augmented samples. Total size: {len(combined_df)}")
            return combined_df

        return df

    def get_lr_schedule(self):
        """Create a learning rate schedule for better convergence"""
        def lr_schedule(epoch, lr):
            if epoch < 5:
                return lr
            elif epoch < 10:
                return lr * 0.8
            else:
                return lr * 0.5

        return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    def train(self, epochs=15, batch_size=32):
        """Train the plagiarism detection model with enhanced techniques"""
        logger.debug("Training plagiarism detection model with enhanced features")

        # Load and preprocess data
        df = self.load_and_preprocess_data()

        # Augment training data for better generalization
        df = self.augment_training_data(df)

        # Train or load Word2Vec model
        try:
            self.word2vec_model = Word2Vec.load(self.word2vec_path)
            print("Loaded existing Word2Vec model")
        except Exception:
            print("Training new Word2Vec model")
            self.train_word2vec_batched(df)

        # Prepare data with enhanced similarity features
        X_text_train, X_text_test, X_sim_train, X_sim_test, y_train, y_test = self.prepare_data_for_training_with_similarity(df)

        # Clear memory
        del df
        gc.collect()

        # Build the enhanced combined model
        self.model = self.build_combined_model(
            X_text_train.shape,
            (X_sim_train.shape[0], X_sim_train.shape[1])
        )

        # Calculate class weights to handle any class imbalance
        class_counts = np.bincount(y_train.astype(int))
        total = np.sum(class_counts)
        class_weights = {i: total / count for i, count in enumerate(class_counts)}
        print(f"Using class weights: {class_weights}")

        # Set up callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            ),
            # Add model checkpoint callback to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath='plagiarism_model/best_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                save_weights_only=False  # Save the full model
            ),
            self.get_lr_schedule()
        ]

        # Train the model with class weights
        logger.debug("Training model with enhanced features and class weights")
        history = self.model.fit(
            [X_text_train, X_sim_train], y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([X_text_test, X_sim_test], y_test),
            callbacks=callbacks,
            class_weight=class_weights
        )

        print("Using model with best weights from callbacks")

        # Evaluate the model
        print("Evaluating model...")
        results = self.model.evaluate([X_text_test, X_sim_test], y_test)

        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1] * 100:.2f}%")
        print(f"Test Precision: {results[2] * 100:.2f}%")
        print(f"Test Recall: {results[3] * 100:.2f}%")
        print(f"Test AUC: {results[4]:.4f}")

        self.model.save_weights('plagiarism_model/best_model.weights.h5')
        print("Model weights saved")

        # Then continue with the existing code
        model_path = 'plagiarism_model/best_model.keras'
        self.save_model(model_path)

        # Plot training history
        try:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')

            plt.tight_layout()
            plt.savefig('plagiarism_model/training_history.png')
            plt.close()
            print("Training history plot saved to 'plagiarism_model/training_history.png'")
        except Exception as e:
            print(f"Could not create training plot: {str(e)}")

        return self.model, self.word2vec_model, self.max_length

    def save_model(self, model_path="plagiarism_model/plagiarism_model.keras"):
        """Save the trained model in the modern format"""
        logger.debug(f"Saving model to {model_path}")
        if self.model:
            # Make sure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Enable unsafe deserialization for Lambda layer
            tf.keras.utils.disable_interactive_logging()
            tf.keras.config.enable_unsafe_deserialization()

            # Updated save options - removed the unsupported 'save_traces' argument
            save_options = {
                'include_optimizer': True,
                'save_format': 'keras'
            }

            try:
                # Save model using the Keras format with proper options
                self.model.save(model_path, **save_options)

                # Save metadata about the model
                metadata = {
                    "max_length": self.max_length,
                    "word2vec_path": self.word2vec_path,
                    "date_trained": str(pd.Timestamp.now())
                }

                with open(os.path.join(os.path.dirname(model_path), "model_metadata.json"), "w") as f:
                    json.dump(metadata, f)

                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Warning: Could not save model in .keras format: {str(e)}")

                # Fallback to SavedModel format if .keras format fails
                alternative_path = os.path.join(os.path.dirname(model_path), "saved_model")
                print(f"Attempting to save in SavedModel format to {alternative_path}")
                self.model.save(alternative_path)
                print(f"Model saved as SavedModel format to {alternative_path}")

            finally:
                tf.keras.utils.enable_interactive_logging()
        else:
            print("No model to save. Train a model first.")


if __name__ == "__main__":
    # Set lower memory TensorFlow options
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    trainer = PlagiarismModelTrainer(batch_size=1000)
    model, word2vec_model, max_length = trainer.train(epochs=10, batch_size=64)
    trainer.save_model()
