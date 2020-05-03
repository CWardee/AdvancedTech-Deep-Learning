from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import numpy as numpyArray
from PIL import Image
from keras.preprocessing import image
from tqdm import tqdm
import os


# Classifier variables
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])





from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        "Data/training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary")

test_set = train_datagen.flow_from_directory(
        "Data/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary")

classifier.fit_generator(
        training_set,
        steps_per_epoch=15,
        epochs=20,
        validation_data=test_set,
        validation_steps=4)


GENERATE_RES = 2 # (1=32, 2=64, 3=96, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3
DATA_PATH = 'data'


# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16
SAVE_FREQ = 100

# Size vector to generate images from
SEED_SIZE = 200
EPOCHS = 34
EPOCHS = EPOCHS * 100
BATCH_SIZE = 16



currentDataSet = 0
print(f"Will generate {GENERATE_SQUARE}px square images.")


# import data set ////////////////////////////////////////////////////////////
dataSet_binaryPath = os.path.join(DATA_PATH,f'data_set')

print(f"Looking for file: {dataSet_binaryPath}")

if not os.path.isfile(dataSet_binaryPath):
  print("Loading training images...")

  dataSetArray = []
  if currentDataSet == 0:
      dataSetLength = os.path.join(DATA_PATH,'dirt_data_set')
  elif currentDataSet == 1:
      dataSetLength = os.path.join(DATA_PATH,'frilly_data_set')
      
  for filename in tqdm(os.listdir(dataSetLength)):
      
      path = os.path.join(dataSetLength,filename)
      image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)
      
      dataSetArray.append(numpyArray.asarray(image))
  dataSetArray = numpyArray.reshape(dataSetArray,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))
  dataSetArray = dataSetArray / 127.5 - 1.


  print("Saving training image binary...")
  numpyArray.save(dataSet_binaryPath,dataSetArray)
  
else:
  print("Loading previous training pickle...")
  dataSetArray = numpyArray.load(dataSet_binaryPath)



# define the standalone generator model //////////////////////////////////////
def define_generator(seed_size, channels):
    model = Sequential()

# layer --------    
    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(Reshape((4,4,256)))
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
  
# layer --------
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

# layer --------
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    # Output resolution, additional upsampling
    for i in range(GENERATE_RES):
      model.add(UpSampling2D())
      model.add(Conv2D(128,kernel_size=3,padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    input = Input(shape=(seed_size,))
    generated_image = model(input)

    return Model(input,generated_image)



# define the standalone discriminator model //////////////////////////////////
def define_discriminator(image_shape):
    model = Sequential()
    
# layer --------
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    
# layer --------
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
# layer --------
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
       
# layer --------
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

# layer --------
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

# layer --------
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

# layer --------
    input_image = Input(shape=image_shape)

    validity = model(input_image)

    return Model(input_image, validity)



def save_images(cnt,noise):
  from keras.preprocessing import image
  image_array = numpyArray.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=numpyArray.uint8)
  
  generated_images = generator.predict(noise)
  current_image = generator.predict(noise)
  
  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH)
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"image_to_test.png")
  im = Image.fromarray(image_array)
  im.save(filename)
  

  imageToTest = image.load_img("data/image_to_test.png", target_size = (64, 64))
  imageToTest = image.img_to_array(imageToTest)
  imageToTest = numpyArray.expand_dims(imageToTest, axis = 0)
  result = classifier.predict(imageToTest)
  training_set.class_indices
  
  if result[0][0] >= 0.5:
        output_path = os.path.join(DATA_PATH,'dirt_output_file')
        prediction = "Dirt has been generated"
        
        print(prediction)
        
        filename = os.path.join(output_path,f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)   
    
        
  else:
        output_path = os.path.join(DATA_PATH,'frilly_output_file')
        prediction = "Grass has been generated"
        
        print(prediction)
        
        filename = os.path.join(output_path,f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)   

      
     
      
      
      
      
  
  
  
image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)
optimizer = Adam(1.5e-4,0.5) # learning rate and momentum adjusted from paper

discriminator = define_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
generator = define_generator(SEED_SIZE,IMAGE_CHANNELS)

random_input = Input(shape=(SEED_SIZE,))

generated_image = generator(random_input)

discriminator.trainable = False

validity = discriminator(generated_image)

combined = Model(random_input,validity)
combined.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])




y_real = numpyArray.ones((BATCH_SIZE,1))
y_fake = numpyArray.zeros((BATCH_SIZE,1))

fixed_seed = numpyArray.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))



GRASS_COUNT = 1
cnt = 1
maxDataSetNumber = 2
dataSetNumber = 0

for dataSetNumber in range(maxDataSetNumber):
                         
    for epoch in range(EPOCHS):
        
        dataSetLength = os.path.join(DATA_PATH,'collected_data_set')
        
        idx = numpyArray.random.randint(0,dataSetArray.shape[0],BATCH_SIZE)
        x_real = dataSetArray[idx]
    
        # Generate some images
        seed = numpyArray.random.normal(0,1,(BATCH_SIZE,SEED_SIZE))
        x_fake = generator.predict(seed)
    
        # Train discriminator on real and fake
        discriminator_metric_real = discriminator.train_on_batch(x_real,y_real)
        discriminator_metric_generated = discriminator.train_on_batch(x_fake,y_fake)
        discriminator_metric = 0.5 * numpyArray.add(discriminator_metric_real,discriminator_metric_generated)
        
        # Train generator on Calculate losses
        generator_metric = combined.train_on_batch(seed,y_real)
        
        # Time for an update?
        if epoch % SAVE_FREQ == 0:
            
            
            save_images(cnt, fixed_seed)
            cnt += 1
            print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
            
    generator.save(os.path.join(DATA_PATH,"gan_network.h5"))



