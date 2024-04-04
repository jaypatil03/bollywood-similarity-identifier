from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

detector = MTCNN()

#load the image
sample_image = cv2.imread('sample/salman_dup.jpg')
results = detector.detect_faces(sample_image)

x,y,width,height = results[0]['box']

face = sample_image[y:y+height, x:x+width]

#Extract the features
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array, axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
# print(result)
# print(result.shape)

#find the cosine distance of current image with all the images in the dataset
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)),reverse=True, key=lambda x: x[1])[0][0]

temp_image = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_image)
cv2.waitKey(0)