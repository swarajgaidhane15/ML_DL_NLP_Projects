# pretrained model by researches at Oxford that will give us the features (2048 in this model)
from keras_vggface import VGGFace

model = VGGFace(model='resnet50',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
                )
