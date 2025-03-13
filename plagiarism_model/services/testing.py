from nltk.tokenize import word_tokenize  # type: ignore
from gensim.models import Word2Vec  # type: ignore
import re
import nltk  # type: ignore
import json
import os
import docx  # type: ignore
import PyPDF2  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional  # type: ignore
import nbformat  # type: ignore


class PlagiarismDetector:
    def __init__(self, model_dir="plagiarism_model",
                 model_path=None,
                 word2vec_path=None,
                 max_length=None):
        """Initialize the plagiarism detector with model paths and settings"""

        # Try to load metadata first if available
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.max_length = metadata.get("max_length", 20)
                    self.word2vec_path = metadata.get("word2vec_path", os.path.join(model_dir, "word2vec_model.bin"))
            except Exception as e:
                print(f"Warning: Could not load metadata: {str(e)}")

        # Override with explicit parameters if provided
        self.max_length = max_length or getattr(self, 'max_length', 20)
        self.word2vec_path = word2vec_path or getattr(self, 'word2vec_path', os.path.join(model_dir, "word2vec_model.bin"))

        # Determine model path - try SavedModel format if available
        self.model_dir = model_dir
        if model_path:
            self.model_path = model_path
        else:
            # Try different model locations
            saved_model_path = os.path.join(model_dir, "saved_model")
            weights_path = os.path.join(model_dir, "best_model.weights.h5")
            keras_path = os.path.join(model_dir, "best_model.keras")

            if os.path.exists(saved_model_path):
                self.model_path = saved_model_path
            elif os.path.exists(weights_path):
                self.model_path = weights_path
            elif os.path.exists(keras_path):
                self.model_path = keras_path
            else:
                self.model_path = os.path.join(model_dir, "plagiarism_model.keras")

        # Download required NLTK resources
        nltk.download('punkt', quiet=True)

        # Load models
        self._load_models()

    def _recreate_model_architecture(self, vector_size=100):
        """Recreate the model architecture to match the original training model"""
        print("Recreating model architecture to match the original training model")

        # Text input branch
        text_input = tf.keras.Input(shape=(self.max_length * 2, vector_size), name='text_input')

        # Process text with Bidirectional LSTM
        x_text = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5))
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

        # Global average pooling instead of Lambda layer
        x_text = tf.keras.layers.GlobalAveragePooling1D()(x_text)

        x_text = Dense(
            64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)
        )(x_text)
        x_text = BatchNormalization()(x_text)

        # Similarity features branch - expanded for the 6 features
        sim_input = tf.keras.Input(shape=(6,), name='similarity_input')
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

        # Compile model with the same settings
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        return model

    def _load_models(self):
        """Load the required models for plagiarism detection"""
        try:
            # Load Word2Vec model
            print(f"Loading Word2Vec from: {self.word2vec_path}")
            self.word2vec_model = Word2Vec.load(self.word2vec_path)

            # Get vector size from the Word2Vec model
            vector_size = self.word2vec_model.vector_size

            # Create and compile model architecture based on the original architecture
            self.model = self._recreate_model_architecture(vector_size)

            # Try to load weights if available (instead of full model)
            try:
                # Try different weight loading approaches based on file extension
                if os.path.exists(self.model_path):
                    if self.model_path.endswith('.h5'):
                        print(f"Loading model weights from: {self.model_path}")
                        self.model.load_weights(self.model_path)
                    elif os.path.isdir(self.model_path):
                        print(f"Loading model from SavedModel directory: {self.model_path}")
                        temp_model = tf.keras.models.load_model(
                            self.model_path,
                            custom_objects={
                                'sum_over_axis': lambda x: tf.reduce_sum(x, axis=1),
                                'sum_output_shape': lambda input_shape: (input_shape[0], input_shape[2])
                            }
                        )
                        self.model.set_weights(temp_model.get_weights())
                    else:
                        print("Model weights not found, using untrained model")
                else:
                    print("Model path doesn't exist, using untrained model")
            except Exception as e:
                print(f"Warning: Could not load model weights: {str(e)}")
                print("Using untrained model (predictions will be random)")

            print("Models loaded successfully")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def extract_text_from_file(self, file_path):
        """Extract text content from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

        elif file_extension == ".pdf":
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                return text

        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
            return text

        elif file_extension == ".ipynb":
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                text = ""
                for cell in notebook.cells:
                    if cell.cell_type == "markdown":
                        text += cell.source + " "
                    elif cell.cell_type == "code":
                        text += "".join(cell.source) + " "
                return text
        else:
            raise ValueError("Unsupported file format! Please upload .txt, .pdf, .docx or .ipynb")

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

    def text_to_embedding(self, text):
        """Convert text to embedding vectors using Word2Vec model"""
        # Preprocess text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text)

        # Create embedding matrix
        embedding_matrix = np.zeros((self.max_length, self.word2vec_model.vector_size))

        # Compute document-level features
        if words:
            # Fill embedding matrix with individual word vectors
            for i, word in enumerate(words[:self.max_length]):
                if word in self.word2vec_model.wv:
                    word_vector = self.word2vec_model.wv[word]

                    # Combine word vector with position information
                    position_factor = 1.0 - (i / self.max_length) * 0.2  # Position weighting
                    embedding_matrix[i] = word_vector * position_factor

        return embedding_matrix

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

    def calculate_similarity_features(self, source_text, plagiarized_text):
        """Calculate similarity features between two texts"""
        # Clean and tokenize texts
        source_tokens = self.clean_text(source_text)
        plagiarized_tokens = self.clean_text(plagiarized_text)
        source_tokens_set = set(source_tokens)
        plagiarized_tokens_set = set(plagiarized_tokens)

        if len(source_tokens_set) == 0 or len(plagiarized_tokens_set) == 0:
            return np.zeros(6)  # Return zeros if either text is empty

        # Jaccard similarity
        jaccard = len(source_tokens_set.intersection(plagiarized_tokens_set)) / len(source_tokens_set.union(plagiarized_tokens_set))

        # Containment measure
        containment = len(source_tokens_set.intersection(plagiarized_tokens_set)) / len(source_tokens_set)

        # Length ratio
        length_ratio = min(len(source_text), len(plagiarized_text)) / max(len(source_text), len(plagiarized_text)) if max(len(source_text), len(plagiarized_text)) > 0 else 0

        # N-gram features
        bigram_sim, trigram_sim = self.calculate_ngram_features(source_tokens, plagiarized_tokens)

        # Add longest common subsequence ratio feature
        lcs_ratio = self.calculate_lcs_ratio(source_tokens, plagiarized_tokens)

        # Return all features as a numpy array
        return np.array([[jaccard, containment, length_ratio, bigram_sim, trigram_sim, lcs_ratio]])

    def check_plagiarism(self, source, plagiarized):
        """
        Compare two texts and detect plagiarism using the dual-input model architecture

        Args:
            source (str): The original text
            plagiarized (str): The text to check for plagiarism

        Returns:
            dict: Result containing plagiarism verdict and confidence score
        """
        # Generate text embeddings
        source_embedding = self.text_to_embedding(source)
        plagiarized_embedding = self.text_to_embedding(plagiarized)

        # Concatenate embeddings for the text input
        text_input = np.concatenate([source_embedding, plagiarized_embedding], axis=0)
        text_input = text_input.reshape(1, text_input.shape[0], text_input.shape[1])

        # Calculate similarity features for the similarity input
        similarity_input = self.calculate_similarity_features(source, plagiarized)

        # Make prediction using the dual-input model
        prediction = self.model.predict([text_input, similarity_input], verbose=0)[0][0]

        # Return both the verdict and the confidence score
        return {
            "verdict": "Plagiarism Detected" if prediction >= 0.5 else "No Plagiarism",
            "confidence": float(prediction),
            "is_plagiarized": bool(prediction >= 0.5),
            "similarity_score": float(prediction)  # Same as confidence, but more descriptive name
        }

    def check_files_for_plagiarism(self, source_file_path, plagiarized_file_path):
        """Compare two files and check for plagiarism"""
        source_text = self.extract_text_from_file(source_file_path)
        plagiarized_text = self.extract_text_from_file(plagiarized_file_path)

        return self.check_plagiarism(source_text, plagiarized_text)


if __name__ == "__main__":
    # Example usage
    detector = PlagiarismDetector()

    # Command-line interface
    mode = input("Choose input method (1: User input, 2: File upload): ")

    if mode == "1":
        sample_source_text = input("Enter the source text: ")
        sample_plagiarized_text = input("Enter the suspected plagiarized text: ")

        result = detector.check_plagiarism(sample_source_text, sample_plagiarized_text)
        print(f"\nVerdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Similarity Score: {result['similarity_score']:.2f}")

    elif mode == "2":
        file1 = input("Enter the path of the source file: ")
        file2 = input("Enter the path of the suspected plagiarized file: ")

        result = detector.check_files_for_plagiarism(file1, file2)
        print(f"\nVerdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Similarity Score: {result['similarity_score']:.2f}")

    else:
        print("Invalid choice! Please enter 1 or 2.")
