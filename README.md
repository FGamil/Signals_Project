# **Signals Project**
## First: Data description 

The [Dataset](https://ieee-dataport.org/open-access/oculusgraphy-pediatric-and-adults-electroretinograms-database) was gathered from both adults and pediatric individuals.
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

## Second: Data preprocessing
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
*** Assembling a new dataframe
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

To be able to work with the signals we gathered, first we needed to transform them from time domain to frequency domain using ***Ricker transform***.

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





