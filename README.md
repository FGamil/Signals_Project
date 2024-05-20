# **Signals Project**
## Firstly: Data description 

The [Dataset](https://doi.org/10.21227/y0fh-5v04) was gathered from both adults and pediatric individuals.
Total of 425 patients were tested according to different protocols:

| Protocol | N. Adults | N. pediatrics |
| :---: | :---: | :---: |
| `Scotopic 2.0 ERG Response` | 23 | 51 |
| `Maximum 2.0 ERG Response` | 42 | 80 |
| `Photopic 2.0 ERG Response` | 32 | 74 |
| `Photopic 2.0 EGR Flicker Response` | 38 | 63 | 
|`Scotopic 2.0 ERG Oscillatory Potentials`| 13 | 7 |

**The data consists of :**
- The patient’s age
- The Diagnosis 
- The graph representing the response 
- The potential difference after each half millisecond

Working on this project, only Scotopic, Maximum & Photobic 2.0 ERG Response where taken into consideration.
You'll find the data uploaded to the repository at [01 Appendix 1](https://github.com/FGamil/Signals_Project/blob/main/01%20Appendix%201.xlsx).

## Secondly: Data preprocessing
### Dropping and redefining data
The data we previously referred to weren't appropriately organized to be processed. Therfor, we organized them in a dataframe where we only added the data needed in processing and neglected the rest.

```
#Example using Scotobic 2.0 Response
SR = pd.read_excel("/kaggle/input/electroretinogram/01 Appendix 1.xlsx", sheet_name='Scotopic 2.0 ERG Response')
SR.drop(columns = SR.columns[80:], inplace=True) #Here we removed all the cells that have no value

```
After dropping the empty cells, we redefined the diagnosis classes into **normal** and **abnormal** to make it easier to understand and more efficiently classified. 

```
mapping = {'The functional activity of the central and peripheral parts of the retina in both eyes is preserved.':'abnormal','Pupil is narrow.':'abnormal','The functional activity of the central and peripheral parts of the retina in both eyes is preserved.':'abnormal','Registration of ERG with a narrow pupil.':'abnormal','Registration of the ERG of a narrow pupil from the skin of the eyelid.':'abnormal','ERG pupil is narrow.':'abnormal','Registration of ERG from the skin of the eyelid, the pupil is narrow.':'abnormal','Registration of ERG with a narrow pupil from the skin of the eyelid.':'abnormal','The functional activity of the central and peripheral parts of the retina in both eyes is preserved.':'abnormal','High level of interference due to increased physical activity of the child.':'abnormal','The functional activity of the retina is preserved.':'normal','The functional activity of the retina at the OU is preserved.': 'normal','The functional activity of the retina is preserved.':'normal','No negative dynamics compared to previous data.': 'normal','The functional activity of the retina is preserved.': 'normal','The functional activity of the central and peripheral parts of the retina in both eyes is preserved.':'normal','No diagnosis.': 'normal','The signal is within normal limits.': 'normal','The functional activity of the retina is preserved, corresponds to the age norm symmetrically on the OU.': 'normal','The functional activity of the retina in both eyes is preserved.': 'normal','The amplitude-time characteristics of the EP in a homogeneous field correspond to the norm.': 'normal','The functional activity of the retina in both eyes is preserved. High level of interference due to increased motor activity of the child.': 'normal','The functional activity of the retina at the OU is preserved. No macular pathology was revealed.': 'The functional activity of the central and peripheral parts of the retina in both eyes is preserved.','No diagnosis.': 'normal','Electrogenesis of the central parts of the retina normally does not exclude dystrophy of the rod apparatus of the retina with a favorable prognosis of the course.': 'normal','The functional activity of the central and peripheral parts of the retina in both eyes is preserved.':'normal','Correct configuration potentials. The amplitude-time characteristics of the a- and b-waves of the rod maximal and cone responses correspond to the norm.': 'normal', 'Perhaps the reason for the violation of twilight vision is hypovitaminosis.': 'normal' ,'Hereditary cone dystrophy.': 'abnormal','Retinal rod dystrophy with a favorable prognosis is not excluded.': 'abnormal','Decreased the amplitude of the b-wave of the maximum and rod responses.': 'abnormal','Electrogenesis of the peripheral parts of the retina in the norm, the electrogenesis of the central parts of the retina is sharply impaired. Perhaps there is hereditary cone dystrophy.': 'abnormal','Reduced functional activity of the retina changes at the level of the outer and middle layers in the central and peripheral parts.': 'abnormal','Combination of congenital myopia and congenital hemeralopia.': 'abnormal','OD - outside and in the upper section of the old chorioretinal focus.': 'abnormal','Rod dystrophy of the retina with a favorable prognosis.': 'abnormal','The functional activity of the peripheral parts of the retina is normal, the functional activity of the central parts of the retina is moderately reduced.': 'abnormal','Hereditary rod dystrophy of the retina is possible.': 'abnormal','Central dystrophy with cone-rod dysfunction is possible.': 'abnormal','The preservation of the retinal electrogenesis from its central and peripheral parts, as well as the preservation of conductivity along the pathways of the visual analyzer, can be expected to improve visual functions after restoration of the transparency of the OS optical media.': 'abnormal','Retinal cone-rod dystrophy.': 'abnormal', 'Pronounced organic changes in the outer and inner layers in the center and periphery of the retina.': 'abnormal', 'Moderate pronounced change in the outer and middle layers of the central and peripheral parts of the retina of the right eye.': 'abnormal', 'Decrease in electrogenesis and functional activity of the retina.': 'abnormal', 'Retinal rod dystrophy with a favorable prognosis is possible.': 'abnormal', 'Signs of moderate disturbances in the electrogenesis of the central parts of the retina in the left eye.': 'abnormal', 'OU - maximum ERG b-wave reduction. Central dystrophy with cone-rod dysfunction is possible.': 'abnormal', 'The functional activity of the retina of the left eye is protected by a moderate decrease in the functional activity of the retina of the right eye.': 'abnormal', 'The functional activity of the retina OD is preserved on the OS and is moderately reduced (changes at the level of the outer and middle layers of the retina in the central and peripheral regions).': 'abnormal', 'The functional activity of the central parts of the retina is moderately reduced.': 'abnormal', 'Changes in the bioelectrical activity of the retina against the background of high myopia.': 'abnormal', 'Hereditary rod dystrophy of the retina is not excluded.': 'abnormal', 'Decrease of the B-wave amplitude by 3 times on OD, 2 times on OS.': 'abnormal', 'OU - a pronounced decrease in the b-wave of the maximum response is reduced.': 'abnormal'}
MR.replace(mapping, inplace=True)
PR.replace(mapping, inplace=True)
SR.replace(mapping, inplace=True)
```
Now we got the ,for instance, Scotopic 2.0 Response data looking like that:
<div>
<img src="https://github.com/FGamil/Signals_Project/blob/main/SR.PNG"width=700 heigth=700>   
</div>

### Assembling a new dataframe
So, now to assemble a dataframe that contains the data needed for processing, which are only the diagnosis, patient ID and signal type, we use the coming code:

```
patiant_Diagnosis_M = MR.iloc[4 , :107].tolist()
patiant_Diagnosis_P = PR.iloc[4 , :95].tolist()
patiant_Diagnosis_S = SR.iloc[4 , :73].tolist()

patiant_id_M = MR.iloc[3 , :107].tolist()
patiant_id_P = PR.iloc[3 , :95].tolist() 
patiant_id_S = SR.iloc[3 , :73].tolist()

M_R =['MR']*107
P_R =['PR']*95
S_R =['SR']*73

# Create a dictionary from the two lists
dict = {'Patient_id': patiant_id_M+patiant_id_P+patiant_id_S, 'Label': patiant_Diagnosis_M+patiant_Diagnosis_P+patiant_Diagnosis_S ,'signal_type':M_R+P_R+S_R}

# Create a new dataframe from the dictionary
df = pd.DataFrame(dict)
```
After running this code, we get a dataframe that consists of the data of **299** patients from which **109** are abnormal and **190** are normal. 
Then, we define to new directories named "signals" and "augmented_images" to save the images we're going to get out of the signals we have. 

```
!mkdir signals
!mkdir augmented_images
```
### Wavelet transformation 

To be able to work with the signals we gathered, first we needed to transform them into a both time and frequency representation through ***Ricker wavelet transform (Mexican hat)***.
> The output of applying this wavelet transform are gray-scale scalogram images.
```
#Example on SR signal transformation 
signal_SR=[]
signal_SR_id=[]
for j in range(0,2):
    for i in range (1,74):
        signal = np.array(SR.iloc[21: ,i].tolist())  
        signal_values = signal[np.logical_not(np.isnan(signal))]
        k = repeat_val(len(signal_values))

        signal_values = np.tile(signal_values.tolist()+[0]*0,k-j)
          
        coefficients, frequencies = pywt.cwt(signal_values, scales=np.arange(1, 128), wavelet='mexh')
        plt.figure(figsize=(5.12, 5.12))
        plt.imshow(abs(coefficients) ** 2, cmap='gray', aspect='auto',vmax=abs(coefficients).max(), vmin=-abs(coefficients).min())
        plt.axis('off')
        plt.savefig(os.path.join(path, f'SR_{i}_{j}_f.png'))
        plt.close()
        signal_SR.append(path + f'SR_{i}_{j}_f.png')
        signal_SR_id.append(f'SR_{i}_{j}_f')

```
Note that in order to expand the available data we used the function **repeat_val**
```
def repeat_val(x):
    if x<200 : k=4 
    elif x>200 and x<=400 : k=3 
    elif x>400 and x<=550 : k=2 
    return k
```
to get the signal size and calculate the number of times we're going to repeat each signal through the function **np.tile()**.
Since the data were repeated, we needed to change the labels to deffrentiate between the original and the replicate data.

```
df = pd.DataFrame({'Label': np.tile(df['Label'],2)})
#The resulting data are then grouped in 2 lists
df['img_Path'] = signal_MR+signal_PR+signal_SR
df['img_id'] = signal_MR_id+signal_PR_id+signal_SR_id 
```
These previous codes resulted in increasing the data to be **380** normal and **218** abnormal signal images.
The lists are then converted into a **.CSV** file to be able to use it to train the model before data augmentation to compare between the results. The diagnosis are then mapped into 0s and 1s.
```
df.to_csv('data1.csv')
mapping = {'normal': int(0), 'abnormal': int(1)}  
df.replace(mapping, inplace=True)
df = df.reset_index(drop=True)
```

### Balancing the data 

since the normal and abnormal values are imbalanced, we remove a random 150 image of those labeled to be normal. Following this, the resulting datafrawe is saved into a **.csv** file.
```
dropped_normal = df.query('Label == 0').sample(n=150)
df.drop(droped_normal.index, inplace=True)
df=df.reset_index(drop=True)
df.to_csv('Scalogram_grayscale_images.csv')
```

### Data Augmentation 
To every signal image, 5 augmented images are produced. The changes applied to the images are due to rotation, shearing, zooming, horizontal flipping, and changing brightness.
```
datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)
```
Afterwards, the output images ,**1380 normal and 1284 abnormal**, were saved into the augmented_image directory created before. 
```
for n in range (len(df.img_Path)):
    # Loading a sample image 
    img = load_img(df.img_Path[n]) 
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1, ) + x.shape)
    j=0
    # Loop through the 5 augmented images and save each image with its corresponding label
    for i, batch in enumerate(datagen.flow(x, batch_size=1,save_format='png')):
        # Get the augmented image array
        augmented_image = batch[0]
        # Get the corresponding label
        label = labels[j]
        j+=1
        # Save the augmented image with its corresponding label
        img = array_to_img(augmented_image)
        img.save(os.path.join(augmented_folder_path, f'{df.img_id[n]}_{label}.png'))
        paths_aug.append(f'{augmented_folder_path}/{df.img_id[n]}_{label}.png')  
        lables_aug.append(df.Label[n])
        id_aug.append(f'{df.img_id[n]}_{label}')
        if j==5:
            break  
```
Then, the final dataframe is created and converted into a **.csv** file and later on all of the output files created before are zipped and [downloaded](https://drive.google.com/open?id=1-Yg5qg3a5YJ2_lyF2-BKrJC_rgpHINid&usp=drive_fs).

```
dict = {'img_Path': paths_aug+list(df.img_Path), 'Label': lables_aug+list(df.Label), 'img_id': id_aug+list(df.img_id) }
final = pd.DataFrame(dict)

final.to_csv('Scalogram_grayscale_augumented_images.csv')

shutil.make_archive('/kaggle/working/output', 'zip', '/kaggle/working/')
```
## Thirdly: Signal processing
### Preparing data
After finishing the preprocessing part, the output zip file was then uploaded as a dataset to a new notebook to work on. 
After importing the necessary libraries some hyperparameters that will be used later on on the processing are intialized. 

```
img_height=224
img_width=224
BATCH_SIZE = 60
NUM_EPOCHS = 30
```
Since the dataset is placed in a new notebook, the images pathes have consequently changed. Therfore, the paths had to be modified so that they can be used in the processing stage. 
```
img_Path = []
for i in range (len(df_aug.img_id)):
    if i < 2220:
        img_Path.append(f'/kaggle/input/signals-dataset/augmented_images/{df_aug.img_id[i]}.png')
    else :    
        img_Path.append(f'/kaggle/input/signals-dataset/signals/{df_aug.img_id[i]}.png')
df_aug['img_Path']=img_Path
```
Then the data were divided into **train, test, and validate** with the ratios being 80 : 10 : 10

```
image_train, image_test, label_train, label_test = train_test_split(df_aug.img_Path, df_aug.Label, test_size=0.2, random_state=42,stratify=df_aug['Label'])
image_valid, image_Test, label_valid, label_Test = train_test_split(image_test, label_test, test_size=0.5, random_state=42,stratify=label_test)
```
Those divisions of data were then saved into dataframes to use them in data greneration.

```
train_df_aug = pd.DataFrame({'image_id': image_train, 'label': label_train}).reset_index(drop=True)
valid_df_aug = pd.DataFrame({'image_id': image_valid, 'label': label_valid}).reset_index(drop=True)
test_df_aug = pd.DataFrame({'image_id': image_Test, 'label': label_Test}).reset_index(drop=True)
```
#### train_df_aug
| Label | Count | 
| :---: | :---: |
| normal | 1104 |
| abnormal | 1027 |

#### valid_df_aug
| Label | Count | 
| :---: | :---: |
| normal | 138 |
| abnormal | 128 |

#### test_df_aug
| Label | Count | 
| :---: | :---: |
| normal | 138 |
| abnormal | 129 |

then we started preparing the data before it's lastly inserted to the model 
```
#Example on preparing training data
train_generator = ImageDataGenerator(rescale=1./255.).flow_from_dataframe(
    dataframe=train_df_aug,
    directory='/kaggle/input/signals-dataset',
    x_col='image_id',  # this column contains the filenames
    y_col='label',  # this column contains the class labels
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
```

### CNN Models
#### VGG16
After calling the model, it was modified to take its inputs from the specified data file we generted before and to be accustomed to our specific task.
```
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = tf.keras.layers.Dense(512, activation='relu')(x)

# Add a final softmax layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

# Freeze the layers in base_model
for layer in base_model.layers:
    layer.trainable = False
    
model.summary()
```
The learning rate was then specified and a condition was added for training to stop if the loss was increasing to avoid overfitting and decreasing the processing time while saving the CPU and the memory capacity:

```
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate),loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',    
    min_delta = 0,
    patience = 4,
    verbose = 0, 
    restore_best_weights = True
)
```
The accuracy and loss where then graphed:
```
# Plot training and validation accuracy
plt.plot(model_1.history['binary_accuracy'], label='Training Accuracy')
plt.plot(model_1.history['val_binary_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
<div>
<img src="https://github.com/FGamil/Signals_Project/blob/main/Model%20accuracy.PNG"width=400 heigth=400>   
</div>

```
# Plot training and validation loss
plt.plot(model_1.history['loss'], label='Training Loss')
plt.plot(model_1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
<div>
<img src="https://github.com/FGamil/Signals_Project/blob/main/Model%20loss.PNG"width=400 heigth=400>   
</div>

From then onwardes, we kept changing the threshold till we reached the best confusion matrix:
```
threshold = 0.57
y_pred = (y_pred_[:] > threshold).astype(int)  # Convert to binary predictions
y_true = np.array(test_df_aug.label,int)

cm = confusion_matrix(y_true, y_pred)
# Plot confusion matrix
labels = ['normal', 'retinal dystrophy']  
display_labels = ['True ' + label for label in labels]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.PuBu, values_format=".2f")
plt.title('Confusion Matrix')
plt.show()

model.evaluate(test_generator)[1]
```
<div>
<img src="https://github.com/FGamil/Signals_Project/blob/main/Confusion%20matrix.PNG"width=700 heigth=700>   
</div>
