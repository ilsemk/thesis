import ast
import numpy as np 
import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer

class SVMClassifier:
    def __init__(self):
        # TF-IDF vectorizer, customized to use code-specific tokenization
        self.tfidf = TfidfVectorizer(
            tokenizer = lambda code: self.tokenize_code(code), # Use Python tokenizer
            preprocessor=None,
            token_pattern=None # Disable the default regex tokenizer
        )
        # Vectorizer for AST features
        self.ast_vectorizer = CountVectorizer(
        tokenizer=lambda x: x,     # Identity tokenizer
        preprocessor=lambda x: x,  # Bypass lowercasing and preprocessing
        token_pattern=None
        )
        self.classifier = OneVsOneClassifier(svm.LinearSVC(class_weight='balanced', dual=False))
    
    def extract_ast(self, code):
        '''Extract parent-child not type pairs from the AST of the code'''
        try:
            tree = ast.parse(code)
            pairs = []
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    # Store parent node and child node
                    pairs.append((type(node).__name__, type(child).__name__))
            return pairs
        except Exception:
            # Return empty list if AST parsing fails
            return []

    def tokenize_code(self, code):
        '''Tokenize code using Python's tokenize module to preserve structure'''
        tokens = []
        try:
            lines = code.encode('utf-8').splitlines(keepends=True)
            readline = iter(lines).__next__
            for token in tokenize.tokenize(readline):
                tokens.append(token.string)
            return tokens
        except Exception:
            # Return empty list if tokenization fails
            return []
        

    def features(self, fragments, fit=False):
        '''Extracts features from provided fragment using TF-IDF and AST'''
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(fragments).toarray()
            ast_features = self.ast_vectorizer.fit_transform([self.extract_ast(code) for code in fragments]).toarray()
        else:
            tfidf_matrix = self.tfidf.transform(fragments).toarray()
            ast_features = self.ast_vectorizer.transform([self.extract_ast(code) for code in fragments]).toarray()

        # Combine lexical and structual features into one matrix
        combined = np.hstack([tfidf_matrix, ast_features])
        return combined

    def fit(self, fragments, authors):
        '''Fits classifier with the provided fragments'''
        features = self.features(fragments, fit=True)
        self.classifier.fit(features, authors)
    
    def predict(self, fragments):
        '''Predicts the authors of the fragments'''
        features = self.features(fragments, fit=False)
        return self.classifier.predict(features)