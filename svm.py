import ast
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier

class SVMClassifier:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.classifier = OneVsOneClassifier(svm.LinearSVC(class_weight='balanced', dual=False))
    
    def extract_ast(self, code):
        '''Extract AST features from a code fragment'''
        try:
            tree = ast.parse(code)
            return[len(list(ast.walk(tree)))]
        except Exception:
            return [0]

    def features(self, fragments, fit=False):
        '''Extracts features from provided fragment using TF-IDF and AST'''
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(fragments).toarray()
        else:
            tfidf_matrix = self.tfidf.transform(fragments).toarray()
        
        ast_features = [self.extract_ast(code) for code in fragments]
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