
"""
## 1. Setup
"""
# Set path
LAB_PATH = "dataset"

# import libraries
import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr


"""
## 2. Define the TextureClassifier class
"""

class TextureClassifier:
    def __init__(self):
        # Initialize StandardScaler and classes
        self.scaler = StandardScaler()
        self.classes = ['stone', 'brick', 'wood']

        # SVM hyperparameter grid for tuning
        self.param_grid = {
            'C': [0.1, 1, 10],          # Regularization strength
            'gamma': ['scale', 'auto', 0.01, 0.1],  # Kernel coefficient
            'kernel': ['rbf', 'linear']  # Kernel type
        }

        # Initialize classifiers with GridSearchCV
        self.svm_glcm = GridSearchCV(
            SVC(probability=True),
            param_grid=self.param_grid,
            cv=3,  # 3-fold cross-validation
            verbose=2
        )
        self.svm_lbp = GridSearchCV(
            SVC(probability=True),
            param_grid=self.param_grid,
            cv=3,
            verbose=2
        )

    def load_dataset(self, base_path):
        """Load images with dataset validation"""
        images = []
        labels = []
        image_paths = []

        print("\n🔍 Dataset Summary:")
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                raise ValueError(f"Directory not found: {class_path}")

            files = os.listdir(class_path)
            print(f"- {class_name}: {len(files)} images")

            for img_name in files:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))  # Standardize size
                    images.append(img)
                    labels.append(class_idx)
                    image_paths.append(img_path)
                else:
                    print(f"⚠️ Could not load: {img_path}")

        return np.array(images), np.array(labels), image_paths

    def extract_glcm_features(self, image):
        """GLCM feature extraction with multiple distances/angles"""
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        glcm = graycomatrix(image, distances=distances, angles=angles,
                          symmetric=True, normed=True)

        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            features.extend(graycoprops(glcm, prop).flatten())

        return np.array(features)

    def extract_lbp_features(self, image, radius=3, n_points=24):
        """LBP with configurable parameters"""
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-6  # Avoid division by zero
        return hist

    def prepare_features(self, images):
        """Feature extraction with progress tracking"""
        glcm_features = []
        lbp_features = []

        for i, img in enumerate(images):
            glcm_features.append(self.extract_glcm_features(img))
            lbp_features.append(self.extract_lbp_features(img))

            if (i+1) % 10 == 0:
                print(f"📊 Processed {i+1}/{len(images)} images")

        return np.array(glcm_features), np.array(lbp_features)

    def train(self, train_images, train_labels):
        """Training with hyperparameter tuning"""
        print("\n🔨 Extracting features...")
        glcm_features, lbp_features = self.prepare_features(train_images)

        print("\n⚖️ Scaling GLCM features...")
        glcm_scaled = self.scaler.fit_transform(glcm_features)

        print("\n🎯 Training GLCM classifier (this may take time)...")
        self.svm_glcm.fit(glcm_scaled, train_labels)

        print("\n🎯 Training LBP classifier (this may take time)...")
        self.svm_lbp.fit(lbp_features, train_labels)

        print("\n✅ Training complete!")
        print(f"Best GLCM params: {self.svm_glcm.best_params_}")
        print(f"Best LBP params: {self.svm_lbp.best_params_}")

    def evaluate(self, test_images, test_labels):
        """Comprehensive evaluation"""
        glcm_features, lbp_features = self.prepare_features(test_images)
        glcm_scaled = self.scaler.transform(glcm_features)

        return {
            'GLCM': {
                'accuracy': accuracy_score(test_labels, self.svm_glcm.predict(glcm_scaled)),
                'precision': precision_score(test_labels, self.svm_glcm.predict(glcm_scaled), average='weighted'),
                'confusion': confusion_matrix(test_labels, self.svm_glcm.predict(glcm_scaled))
            },
            'LBP': {
                'accuracy': accuracy_score(test_labels, self.svm_lbp.predict(lbp_features)),
                'precision': precision_score(test_labels, self.svm_lbp.predict(lbp_features), average='weighted'),
                'confusion': confusion_matrix(test_labels, self.svm_lbp.predict(lbp_features))
            }
        }

    def predict(self, image, method='glcm'):
        """Prediction with probability scores"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))

        if method.lower() == 'glcm':
            features = self.extract_glcm_features(image)
            features = self.scaler.transform(features.reshape(1, -1))
            probs = self.svm_glcm.predict_proba(features)[0]
        else:
            features = self.extract_lbp_features(image)
            probs = self.svm_lbp.predict_proba(features.reshape(1, -1))[0]

        return {self.classes[i].capitalize(): float(probs[i]) for i in range(len(self.classes))}


    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for both GLCM and LBP"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # GLCM Confusion Matrix
        sns.heatmap(results['GLCM']['confusion'],
                    annot=True, fmt='d', ax=ax1,
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        ax1.set_title('GLCM Confusion Matrix')

        # LBP Confusion Matrix
        sns.heatmap(results['LBP']['confusion'],
                    annot=True, fmt='d', ax=ax2,
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        ax2.set_title('LBP Confusion Matrix')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()

def classify_texture(image, method):
    """Classify the texture of an image using the selected method."""
    # Default to GLCM if no method is selected
    if method is None or method not in ["GLCM", "LBP"]:
        method = "GLCM"

    # Predict the class of the image
    return classifier.predict(image, method)

"""## 3. Gradio Interface"""

def create_gradio_interface(classifier):
    """Create and launch Gradio interface."""
    iface = gr.Interface(
        fn=classify_texture,
        inputs=[
            gr.Image(type="numpy"),  # Input: Image
            gr.Radio(["GLCM", "LBP"], label="Classification Method", value="GLCM")  # Default to GLCM
        ],
        outputs=gr.Label(num_top_classes=3),  # Output: Predicted class probabilities
        title="Texture Classifier",
        description="Upload an image to classify its texture as Stone, Brick, or Wood",
    )
    return iface

"""## Excution"""

# Main execution
if __name__ == "__main__":
    # Initialize classifier
    classifier = TextureClassifier()

    try:
        # Load and prepare dataset
        print("Loading dataset...")
        images, labels, image_paths = classifier.load_dataset(LAB_PATH)

        # Split dataset
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # Train the classifier
        classifier.train(X_train, y_train)

        # Evaluate the models
        print("\nEvaluating models...")
        results = classifier.evaluate(X_test, y_test) 

        # Print results
        print("\nResults:")
        for method in ['GLCM', 'LBP']:
            print(f"\n{method} Results:")
            print(f"Accuracy: {results[method]['accuracy']:.3f}")
            print(f"Precision: {results[method]['precision']:.3f}")

        # Plot confusion matrices
        classifier.plot_confusion_matrices(results)
        print("\nConfusion matrices have been saved as 'confusion_matrices.png'")

        # Create and launch Gradio interface
        print("\nLaunching Gradio interface...")
        iface = create_gradio_interface(classifier)
        iface.launch()

    except Exception as e:
        print(f"An error occurred: {str(e)}")