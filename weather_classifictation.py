import cv2, random, pickle, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC, LinearSVC


# Import the resized images and transform them to numpy arrays to be consumed by SVC
# Export to pickle file and return the name of the file to be loaded
def create_data():
    data_dir = "Data"
    Categories = ["Category1", "Category2", "Category3", "Category4", "Category5"]
    data = []

    for category in Categories:
        path = os.path.join(data_dir, category)
        label = Categories.index(category)

        for image in os.listdir(path):
            im_path = os.path.join(path, image)
            hurricane_image = cv2.imread(im_path,0)
            final_image = np.array(hurricane_image).flatten()

            data.append([final_image, label])
    
    result = open("data.pickle", "wb")
    pickle.dump(data, result)
    result.close()
    return "data.pickle"


# Prepare the data for the model by shuggling and getting the features and labels
def prepare_data(file_name):
    data_file = open(file_name, "rb")
    data = pickle.load(data_file)
    data_file.close()

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    return x_train, x_test, y_train, y_test


# Create a SVM model that implements the “one-versus-one” approach for multi-class classification
def train_svm_ovo(x_train, y_train, x_test, y_test):
    Categories = ["Category1", "Category2", "Category3", "Category4", "Category5"]

    model = SVC(C=0.85, kernel="poly", gamma="auto", decision_function_shape="ovo")
    model.fit(x_train, y_train)

    # Get the accuracy metrics of the model
    prediction = model.predict(x_test)
    accuracy_ovo = model.score(x_test, y_test)
    confusion_matrix_ovo = confusion_matrix(y_test, prediction)
    f1_score_ovo = f1_score(y_test, prediction, average="weighted")

    # Save the metrics achieved by the model
    acc_metrics = {"accuracy": accuracy_ovo, "f1_score_ovo": f1_score_ovo}
    pd.DataFrame(acc_metrics, index=[0]).to_csv("ovo.csv")
    pd.DataFrame(confusion_matrix_ovo).to_csv("cm_ovo.csv")

    # Save the model
    filename = "ovo_svm_model.sav"
    pickle.dump(model, open(filename, 'wb'))



def train_svm_ovr(x_train, y_train, x_test, y_test):
    Categories = ["Category1", "Category2", "Category3", "Category4", "Category5"]

    model = SVC(C=0.85, kernel="poly", gamma="auto", decision_function_shape="ovr")
    model.fit(x_train, y_train)

    # Get the accuracy metrics of the model
    prediction = model.predict(x_test)
    accuracy_ovr = model.score(x_test, y_test)
    confusion_matrix_ovr = confusion_matrix(y_test, prediction)
    f1_score_ovr = f1_score(y_test, prediction, average="weighted")


    # Save the metrics achieved by the model
    acc_metrics = {"accuracy": accuracy_ovr, "f1_score_ovr": f1_score_ovr}
    pd.DataFrame(acc_metrics, index=[0]).to_csv("ovr.csv")
    pd.DataFrame(confusion_matrix_ovr).to_csv("cm_ovr.csv")


    # Save the model
    filename = "ovr_svm_model.sav"
    pickle.dump(model, open(filename, 'wb'))



def main():
    filename = create_data()
    x_train, x_test, y_train, y_test = prepare_data(filename)
    train_svm_ovo(x_train, y_train, x_test, y_test)
    train_svm_ovr(x_train, y_train, x_test, y_test)



if __name__ == "__main__":
    main()