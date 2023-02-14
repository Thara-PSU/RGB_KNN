import os
import cv2
import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


# Figure of PSU Neurosurgery
def main():
    from PIL import Image
    image_ban = Image.open('images/image2.png')
    st.image(image_ban, use_column_width=False)
    #st.sidebar.image(image_hospital)
if __name__ == '__main__':
    main()

# Set up constants
IMG_SIZE = (128, 128)
NUM_CHANNELS = 3
NUM_CLASSES = 2
K = 5


# Define a function to extract RGB features from an image
def extract_rgb_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)
    features = np.concatenate([r.flatten(), g.flatten(), b.flatten()])
    return features

# Load the data
# Non_germinoma =0, Germinoma =1 ตามลำดับใน enmerate
data = []
labels = []
num_images = 0
for label, subfolder in enumerate(['Non_germinoma', 'Germinoma']):
    folder_path = os.path.join('pineal_tumor', subfolder)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        features = extract_rgb_features(file_path)
        data.append(features)
        labels.append(label)
        num_images += 1

# Print the number of features and images
st.title('Image classification of Pineal Tumor using RGB extraction with k-NN model')
st.subheader('Results of RGB feature extraction')
st.write(f'Total number of features: {len(data[0])}')
st.write(f'Total number of images: {num_images}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a k-NN model on the training set
clf = KNeighborsClassifier(n_neighbors=K)


# Train the  k-NN model
clf.fit(X_train, y_train)


# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() #for specificity calculation



# Compute the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
# Calculate specificity
specificity = tn / (tn + fp)

# Print the performance metrics
st.subheader('Confusion matrix of model developement by test dataset')
st.write(f'Test set accuracy: {accuracy:.3f}')
st.write(f'Test set precision: {precision:.3f}')
st.write(f'Test set recall: {recall:.3f}')
st.write(f'Test set Specificity: {specificity:.3f}')
st.write(f'Test set F1 score: {f1:.3f}')
st.write(f'Test set AUC-ROC score: {auc_roc:.3f}')


# Define the Streamlit app
st.title('Pineal Tumor Classification for unseen image')
st.write('This app classifies images of pineal tumors as either germinoma or non-germinoma.')

# Allow the user to upload an image
uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

# Make a prediction if an image has been uploaded
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_path = 'uploaded_image.jpg'
    cv2.imwrite(img_path, img)
    features = extract_rgb_features(img_path)
    prediction = clf.predict([features])[0]
    if prediction == 0:
        st.subheader('This image is classified as a germinoma.')
    else:
        st.subheader('This image is classified as a non-germinoma.')
