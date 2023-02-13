import os
import dask.bag as db
from PIL import Image
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch



data_directories = {
"Category 1 Hurricane" : len(os.listdir("Raw_Data/Category1")),
"Category 2 Hurricane" : len(os.listdir("Raw_Data/Category2")),
"Category 3 Hurricane" : len(os.listdir("Raw_Data/Category3")),
"Category 4 Hurricane" : len(os.listdir("Raw_Data/Category4")),
"Category 5 Hurricane" : len(os.listdir("Raw_Data/Category5"))}

directories = {
"Category 1 Hurricane" : "Raw_Data/Category1/",
"Category 2 Hurricane" : "Raw_Data/Category2/",
"Category 3 Hurricane" : "Raw_Data/Category3/",
"Category 4 Hurricane" : "Raw_Data/Category4/",
"Category 5 Hurricane" : "Raw_Data/Category5/"}

final_directories = {
"Category 1 Hurricane" : "Data/Category1/",
"Category 2 Hurricane" : "Data/Category2/",
"Category 3 Hurricane" : "Data/Category3/",
"Category 4 Hurricane" : "Data/Category4/",
"Category 5 Hurricane" : "Data/Category5/"}

Categories = ["Category1", "Category2", "Category3", "Category4", "Category5"]



# Plot the cardinality for each data class
def plot_cardinality(data_directories):
    plt.bar(data_directories.keys(), data_directories.values(), width = .5)
    plt.title("Number of Images by Class")
    plt.xlabel('Class Name')
    plt.ylabel('# Images')
    plt.xticks(fontsize=10)
    plt.savefig("classCardinality.jpg")
    plt.show()


# Plot the size of images for each category
def plot_image_size(directories):
    data_list = get_images_size(directories)
    category1 = data_list[0]
    category2 = data_list[1]
    category3 = data_list[2]
    category4 = data_list[3]
    category5 = data_list[4]

    fig, ((axes1, axes2, axes3), (axes4, axes5, axes6)) = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
    category1.plot.scatter(x="Height", y="Width", s=10, c="r", label="Category 1", ax=axes1)
    category2.plot.scatter(x="Height", y="Width", s=10, c="b", label="Category 2", ax=axes2)
    category3.plot.scatter(x="Height", y="Width", s=10, c="g", label="Category 3", ax=axes3)
    category4.plot.scatter(x="Height", y="Width", s=10, c="k", label="Category 4", ax=axes4)
    category5.plot.scatter(x="Height", y="Width", s=10, c="y", label="Category 5", ax=axes5)
    fig.delaxes(axes6)
    plt.savefig("height_width.jpg")
    plt.show()


# Get the smallest image size, to resize all the rest of the images
def get_smallest_size(directories):
    minimum_list = []

    data_list = get_images_size(directories)
    for i in range(len(data_list)):
        min_height = data_list[i]["Height"].min()
        min_width  = data_list[i]["Width"].min()
        minimum_list.append((min_height, min_width))

    return list(map(min, zip(*minimum_list)))


# Get the count of images of each size for each category
def get_images_size(directories):
    list_dfs = []

    for n,d in directories.items():
        filepath = d
        filelist = [filepath + f for f in os.listdir(filepath)]
        dims = db.from_sequence(filelist).map(get_image_dimensions)
        dims = dims.compute()
        dim_df = pd.DataFrame(dims, columns=['Height', 'Width'])
        sizes = dim_df.groupby(['Height', 'Width']).size().reset_index().rename(columns={0:'count'})
        list_dfs.append(sizes)

    return list_dfs


# Get the dimensions of an Image
def get_image_dimensions(file):
    im = Image.open(file)
    arr = np.array(im)
    h,w,d = arr.shape
    return h,w


# Update the images size to the smallest possible, and create the directory 
def image_resize(directories, categories):
    final_size = get_smallest_size(directories)

    os.makedirs("Data")
    for category in categories:
        os.makedirs("Data/{0}".format(category))
    
    for category in list(directories.values()):
        for files in os.listdir(category):
            image = category+files
            im = Image.open(image).resize(final_size)
            im.save("Data/{0}/resized_{1}.jpg".format(category.replace("Raw_Data/", ""), files))

def image_average(directories, categories):
    os.makedirs("Average_Data")
    for category in categories:
        os.makedirs("Average_Data/{0}".format(category))

    for category in list(directories.values()):

        final_size = get_smallest_size(directories)
        number_of_images = len(fnmatch.filter(os.listdir(category), '*.jpg'))
        color_array = np.zeros((final_size[0], final_size[1], 3), np.float)

        for file in os.listdir(category):
            image_array = np.array(Image.open(category+file),dtype=np.float)
            color_array = color_array + image_array 

        color_array = color_array / number_of_images
        color_array = np.array(np.round(color_array), dtype=np.uint8)
        average_image = Image.fromarray(color_array, mode="RGB")
        average_image.save("{0}average.jpg".format(category.replace("Data/", "Average_Data/")))


def main():
    plot_cardinality(data_directories)
    plot_image_size(directories)
    image_resize(directories, Categories)
    image_average(final_directories, Categories)



if __name__ == "__main__":
    main()