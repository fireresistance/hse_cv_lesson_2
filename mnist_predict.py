
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def load_image(fname):
    img = load_img(fname, grayscale=True, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

def predict_image():
    img = load_image("sample_image.png")
    #model_name = str()
    model = load_model('mnist.h5', compile=False)
    vector = model.predict_classes(img)
    print(vector[0])

predict_image()