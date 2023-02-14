import pickle, os,cv2
import numpy as np

filename_ovo = "ovo_svm_model.sav"
filename_ovr = "ovr_svm_model.sav"
image_path = "Data/Category1/resized_25__.jpg"



# Load the models for demo and give predictions
def demo_models(filename_ovo, filename_ovr, image_path):
    Categories = ["Category1", "Category2", "Category3", "Category4", "Category5"]

    ovo_model = pickle.load(open(filename_ovo, 'rb'))
    ovr_model = pickle.load(open(filename_ovr, 'rb'))
    image = np.array(cv2.imread(image_path,0)).reshape(1,-1)

    prediction_ovo = ovo_model.predict(image)
    prediction_ovr = ovr_model.predict(image)
    print("The prediction using the One over One SVM is: {0}".format(Categories[prediction_ovo[0]]))
    print("The prediction using the One over Rest SVM is: {0}".format(Categories[prediction_ovr[0]]))
    return prediction_ovo, prediction_ovr





def main():
    demo_models(filename_ovo, filename_ovr, image_path)



if __name__ == "__main__":
    main()