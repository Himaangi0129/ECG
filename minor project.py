#!/usr/bin/env python
# coding: utf-8

# # LSTM
# ### step 1 : importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


# In[2]:


df = pd.read_csv("ecg.csv", header = None)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df[140].unique()


# Display the number of examples for each lable:aberrant and normal ECG

# In[9]:


df[140].value_counts()


# ### data visualization

# In[10]:


plt.plot(df.iloc[0,: 140])
plt.xlabel("time[sec]")
plt.ylabel("ECG[mV]")
plt.title("monitoring of fiducial points")
plt.grid(True)
plt.show()


# plotting graphs of normal and abnormal ECG

# In[11]:


abnormal = df[df.loc[:,140] ==0][:10]
normal = df[df.loc[:,140] ==1][:10]
# Create the figure 
fig = go.Figure()
#create a list to display only a single legend 
leg  = [False] * abnormal.shape[0]
leg[0] = True


# Plot training and validation error
for i in range(abnormal.shape[0]):
    fig.add_trace(go.Scatter( x=np.arange(abnormal.shape[1]),y=abnormal.iloc[i,:],name='Abnormal ECG', mode='lines',  marker_color='rgba(255, 0, 0, 0.9)', showlegend= leg[i]))
for j in range(normal.shape[0]):
    fig.add_trace(go.Scatter( x=np.arange(normal.shape[1]),y=normal.iloc[j,:],name='Normal ECG',  mode='lines',  marker_color='rgba(0, 255, 0, 1)', showlegend= leg[j]))
fig.update_layout(xaxis_title="time", yaxis_title="Signal", title= {'text': 'Difference between different ECG', 'xanchor': 'center', 'yanchor': 'top', 'x':0.5} , bargap=0,)
fig.update_traces(opacity=0.5)
fig.show()


# ### data preprocessing  (LSTM)

# In[12]:


# split the data into labels and features 
ecg_data = df.iloc[:,:-1] # features
labels = df.iloc[:,-1] #target value


# In[13]:


ecg_data.head()


# In[14]:


labels.head()


# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


# Normalize the data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
ecg_data = scaler.fit_transform(ecg_data)


# In[17]:


ecg_data


# In[18]:


from scipy.signal import medfilt, butter, filtfilt


# In[19]:


#filtering the raw signals
# Median filtering
ecg_medfilt = medfilt(ecg_data, kernel_size=3)


# In[20]:


# Low-pass filtering
lowcut = 0.05
highcut = 20.0
nyquist = 0.5 * 360.0
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band')
ecg_lowpass = filtfilt(b, a, ecg_data)


# In[21]:


import pywt


# In[22]:


# Wavelet filtering
coeffs = pywt.wavedec(ecg_data, 'db4', level=1)
threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(ecg_data)))
coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
ecg_wavelet = pywt.waverec(coeffs, 'db4')


# In[23]:


# plotting graph of unfiltered and filtered data 


# In[24]:


# Plot original ECG signal
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(ecg_data.shape[0]), y=ecg_data[30], mode='lines', name='Original ECG signal'))
# Plot filtered ECG signals
fig.add_trace(go.Scatter(x=np.arange(ecg_medfilt.shape[0]), y=ecg_medfilt[30], mode='lines', name='Median filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_lowpass.shape[0]), y=ecg_lowpass[30], mode='lines', name='Low-pass filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_wavelet.shape[0]), y=ecg_wavelet[30], mode='lines', name='Wavelet filtered ECG signal'))
fig.show()


# In[25]:


#choosing best filtering data from above 


# In[26]:


#pad the signal with zeroes
def pad_data(original_data,filtered_data):
  # Calculate the difference in length between the original data and filtered data
  diff = original_data.shape[1] - filtered_data.shape[1]
    # pad the shorter array with zeroes
  if diff > 0:
          # Create an array of zeros with the same shape as the original data
      padding = np.zeros((filtered_data.shape[0], original_data.shape[1]))
      # Concatenate the filtered data with the padding
      padded_data = np.concatenate((filtered_data, padding))
  elif diff < 0:
      padded_data = filtered_data[:,:-abs(diff)]
  elif diff == 0:
      padded_data = filtered_data
  return padded_data


# In[27]:


def mse(original_data, filtered_data):
    filter_data = pad_data(original_data,filtered_data)
    return np.mean((original_data - filter_data) ** 2)
# Calculate MSE
mse_value_m = mse(ecg_data, ecg_medfilt)
mse_value_l = mse(ecg_data, ecg_lowpass)
mse_value_w = mse(ecg_data, ecg_wavelet)
print("MSE value of Median Filtering:", mse_value_m)
print("MSE value of Low-pass Filtering:", mse_value_l)
print("MSE value of Wavelet Filtering:", mse_value_w)


# In[28]:


#Based on the MSE value displayed above, wavelet filtering is chosen.


# ### splitting of test and train data 

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train , X_test, y_train, y_test = train_test_split(ecg_wavelet, labels, test_size=0.2, random_state=42)


# In[31]:


import scipy.signal


# ### Feature extraction

# In[32]:


# Initializing an empty list to store the features
features = []
# Extracting features for each sample
for i in range(X_train.shape[0]):
    #Finding the R-peaks
    r_peaks = scipy.signal.find_peaks(X_train[i])[0]
    #Initialize lists to hold R-peak and T-peak amplitudes
    r_amplitudes = []
    t_amplitudes = []
    # Iterate through R-peak locations to find corresponding T-peak amplitudes
    for r_peak in r_peaks:
        # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
        t_peak = np.argmin(X_train[i][r_peak:r_peak+200]) + r_peak
        #Append the R-peak amplitude and T-peak amplitude to the lists
        r_amplitudes.append(X_train[i][r_peak])
        t_amplitudes.append(X_train[i][t_peak])
    # extracting singular value metrics from the r_amplitudes
    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)
    # extracting singular value metrics from the t_amplitudes
    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)
    # Find the time between consecutive R-peaks
    rr_intervals = np.diff(r_peaks)
    # Calculate the time duration of the data collection
    time_duration = (len(X_train[i]) - 1) / 1000 # assuming data is in ms
    # Calculate the sampling rate
    sampling_rate = len(X_train[i]) / time_duration
    # Calculate heart rate
    duration = len(X_train[i]) / sampling_rate
    heart_rate = (len(r_peaks) / duration) * 60
    # QRS duration
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    # extracting singular value metrics from the qrs_durations
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)
    # Extracting the singular value metrics from the RR-interval
    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)
    # Extracting the overall standard deviation
    std = np.std(X_train[i])
    # Extracting the overall mean
    mean = np.mean(X_train[i])
    # Appending the features to the list
    features.append([mean, std, std_qrs, mean_qrs,median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr,median_rr, heart_rate])
# Converting the list to a numpy array
features = np.array(features)


# We have now extracted 19 features from the dataset.
# 
# The shape of this training set after feature extraction is:
# 
# (3998, 19)

# In[33]:


features


# we will extract the features of the test set

# In[34]:


# Initializing an empty list to store the features
X_test_fe = []
# Extracting features for each sample
for i in range(X_test.shape[0]):
    # Finding the R-peaks
    r_peaks = scipy.signal.find_peaks(X_test[i])[0]
    # Initialize lists to hold R-peak and T-peak amplitudes
    r_amplitudes = []
    t_amplitudes = []
    # Iterate through R-peak locations to find corresponding T-peak amplitudes
    for r_peak in r_peaks:
        # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
        t_peak = np.argmin(X_test[i][r_peak:r_peak+200]) + r_peak
        # Append the R-peak amplitude and T-peak amplitude to the lists
        r_amplitudes.append(X_test[i][r_peak])
        t_amplitudes.append(X_test[i][t_peak])
    #extracting singular value metrics from the r_amplitudes
    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)
    #extracting singular value metrics from the t_amplitudes
    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)
    # Find the time between consecutive R-peaks
    rr_intervals = np.diff(r_peaks)
    # Calculate the time duration of the data collection
    time_duration = (len(X_test[i]) - 1) / 1000 # assuming data is in ms
    # Calculate the sampling rate
    sampling_rate = len(X_test[i]) / time_duration
    # Calculate heart rate
    duration = len(X_test[i]) / sampling_rate
    heart_rate = (len(r_peaks) / duration) * 60
    # QRS duration
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    #extracting singular value metrics from the qrs_duartions
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)
    # Extracting the standard deviation of the RR-interval
    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)
      # Extracting the standard deviation of the RR-interval
    std = np.std(X_test[i])
    # Extracting the mean of the RR-interval
    mean = np.mean(X_test[i])
    # Appending the features to the list
    X_test_fe.append([mean, std,  std_qrs, mean_qrs,median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr,median_rr,heart_rate])
# Converting the list to a numpy array
X_test_fe = np.array(X_test_fe)


# In[35]:


X_test_fe


# The shape of the test set after feature extraction is as follows:
# 
# (1000, 19)

# ### Model building and training

# 1) we will reshape the data to make it compatible with the model  ||
# 2) we will create an LSTM model with only 2 layers  ||
# 3) we will train it on the features extracted from the data  ||
# 4) we will make the predictions on the validation/test set  ||

# In[36]:


pip install keras


# In[37]:


pip install tensorflow


# In[38]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score


# In[39]:


# Define the number of features in the train dataframe
num_features = features.shape[1]
# Reshape the features data to be in the right shape for LSTM input
features = np.asarray(features).astype('float32')
features = features.reshape(features.shape[0], features.shape[1], 1)
X_test_fe = X_test_fe.reshape(X_test_fe.shape[0], X_test_fe.shape[1], 1)
# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(features.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(features, y_train, validation_data=(X_test_fe, y_test), epochs=50, batch_size=32)
# Make predictions on the validation set
y_pred = model.predict(X_test_fe)
# Convert the predicted values to binary labels
y_pred = [1 if p>0.5 else 0 for p in y_pred]
X_test_fe = np.asarray(X_test_fe).astype('float32')


# ### Model Evaluation

# 1. Calculating all the metrics

# In[40]:


# calculate the accuracy
acc = accuracy_score(y_test, y_pred)
#calculate the AUC score
auc = round(roc_auc_score(y_test, y_pred),2)
#classification report provides all metrics e.g. precision, recall, etc. 
all_met = classification_report(y_test, y_pred)


#  2. Displaying all the metrics

# In[41]:


# Print the accuracy
print("Accuracy: ", acc*100, "%")
print(" \n")
print("AUC:", auc)
print(" \n")
print("Classification Report:\n",all_met)
print(" \n")


# 3. Calculating and displaying the confusion matrix

# In[42]:


import plotly.express as px


# In[43]:


# Calculate the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_df = pd.DataFrame(conf_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
fig = px.imshow(conf_mat_df, text_auto= True, color_continuous_scale='Blues')
fig.update_xaxes(side='top', title_text='Predicted')
fig.update_yaxes(title_text='Actual')
fig.show()


# ### Plotting the Training and Validation Error

# In[44]:


# Plot training and validation error
fig = go.Figure()
fig.add_trace(go.Scatter( y=history.history['loss'], mode='lines', name='Training'))
fig.add_trace(go.Scatter( y=history.history['val_loss'], mode='lines', name='Validation'))
fig.update_layout(xaxis_title="Epoch", yaxis_title="Error", title= {'text': 'Model Error', 'xanchor': 'center', 'yanchor': 'top', 'x':0.5} , bargap=0)
fig.show()


# ### Result 

# The model achieved a recall value of 0.92 and an AUC score of 0.93, exhibiting its effectiveness with a simple deep-learning architecture. In the healthcare sector, recall is a crucial metric, especially in disease screening, where false negatives can lead to serious consequences, including missed early detection and treatment opportunities. The good recall score in this project highlights the potential for this model to impact disease screening efforts positively.

# # NORMAL AUTOENCODER 
# 

# In[45]:


data = df.loc[:, 0:len(df.columns) -2]
label = df.loc[ :, len(df.columns) -1]


# Time series analysis 

# In[46]:


signal = data.iloc[np.random.randint(0, len(data))].to_list()


# Plotting the time series

# In[47]:


plt.figure( figsize = (15,7))
plt.plot(signal)
plt.title("ECG Signal")
plt.grid(True)
plt.show()


# Plotting Moving Average 

# In[48]:


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


# In[49]:


signal = pd.Series(signal)


# In[50]:


def moving_average(series, n):
    'Calculate average of last n observations'
    return np.average(series[-n: ])


# In[51]:


def plotMovingAverage( series , window, plot_intervals= False, scale=1.96,
                     plot_anomalies=False):
    """
       series - dataframe with timeseries
       window - rolling window size
       plot_intervals - show confidence intervals
       plot_anomalies - show anomalies
    
    """
    rolling_mean = series.rolling(window = window).mean()
    
    plt.figure(figsize =(15, 5))
    plt.title('Moving average\n windoe size ={}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    # plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window: ], rolling_mean[window: ])
        deviation = np.std(series[window: ]- rolling_mean[window: ])
        lower_bond = rolling_mean - (mae + scale* deviation)
        upper_bond = rolling_mean + (mae+scale*deviation)
        plt.plot(upper_bond, 'r--', label ='Upper Bond/Lower Bond')
        plt.plot(lower_bond, 'r--')
        
        # Having the intervals, find abnormal values
        
        if plot_anomalies:
            anomalies = pd.DataFrame(index = series.index, columns = series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, 'ro', markersize =10)
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc ='upper left')
    plt.grid(True)
            


# In[52]:


plotMovingAverage(signal, 4)


# In[53]:


plotMovingAverage(signal, 12) #smoothening


# In[54]:


plotMovingAverage(signal, 4, plot_intervals=True)


# In[55]:


plt.hist(signal)
plt.show()


# ### Splitting Dataset

# In[99]:


x_train, x_test, Y_train, Y_test = train_test_split(data, label, test_size = 0.25, shuffle= True)


# In[100]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ### Defining Model

# Logistic Regression

# In[101]:


lor_model = LogisticRegression(solver='lbfgs', max_iter=1000)
lor_model.fit(x_train, Y_train)

lor_pred = lor_model.predict(x_test)


# In[103]:


print(f"Classification Report of Logistic Regression: \n {classification_report(lor_pred, Y_test)}")


# In[104]:


svm_model = SVC(gamma = 'auto')
svm_model.fit(x_train, Y_train)
svm_pred = svm_model.predict(x_test)


# In[105]:


print(f"Classification Report of SVM: \n {classification_report(svm_pred, Y_test)}")


# ### Confusion Matrix

# In[106]:


lor_cm = confusion_matrix(Y_test, lor_pred)
svm_cm = confusion_matrix(Y_test, svm_pred)

class_names = np.unique(Y_train)

plt.figure(figsize = (30,10))

plt.subplot(1,2,1)
sns.heatmap(lor_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)

plt.subplot(1,2,2)
sns.heatmap(svm_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)

plt.show()


# ### Metrics

# In[107]:


r2_svm = r2_score(Y_test, svm_pred)
r2_lor = r2_score(Y_test, lor_pred)

mae_svm = mean_absolute_error(Y_test, svm_pred)
mae_lor = mean_absolute_error(Y_test, lor_pred)

mse_svm = mean_squared_error(Y_test, svm_pred)
mse_lor = mean_squared_error(Y_test, lor_pred)

msle_svm = mean_squared_log_error(Y_test, svm_pred)
msle_lor = mean_squared_log_error(Y_test, lor_pred)


# In[108]:


metrics_dict = {"SVM" : [r2_svm, mae_svm, mse_svm, msle_svm], "Logistic Regression" : [r2_lor, mae_lor, mse_lor, msle_lor]}


# In[109]:


metrics = pd.DataFrame(metrics_dict)
metrics


# # CNN

# In[83]:


# importing required libraries 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# In[84]:


df.shape


# In[85]:


df.head()


# In[87]:


X = df.iloc[:,:-1] # features
Y = df.iloc[:,-1] #target


# In[88]:


plt.figure(figsize=(15,7))
g = sns.countplot(Y, palette="icefire")
Y.value_counts()


# In[110]:


# Splitting the data 
X_Train, X_Test,Y_Train, Y_Test = train_test_split(X,Y, test_size =0.2, random_state= 42) 


# In[111]:


# standardize the data
scaler = StandardScaler()
X_Train= scaler.fit_transform(X_Train)
X_Test= scaler.transform(X_Test)


# In[112]:


#define a simple CNN model
model = keras.Sequential([
    keras.layers.Reshape((X_Train.shape[1], 1), input_shape=(X_Train.shape[1],)),
    keras.layers.Conv1D(32, 5, activation='relu'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])


# In[113]:


# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[114]:


# train the model
history = model.fit(X_Train, Y_Train, epochs=10, validation_data=(X_Test, Y_Test))


# In[115]:


#evaluate the model
test_loss, test_acc = model.evaluate(X_Test, Y_Test)
print(f'Test accuracy: {test_acc}')


# In[116]:


# make predictions 
y_pred = model.predict(X_Test)
y_pred_classes = np.argmax(y_pred, axis=1)


# In[117]:


# Display confusion matrix and classification report
confusion = confusion_matrix(Y_Test, y_pred_classes)
classification_rep = classification_report(Y_Test, y_pred_classes)


# In[118]:


print('Confusion Matrix:')
print(confusion)

print('Classification Report:')
print(classification_rep)


# In[119]:


# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# In[120]:


# Calculate the confusion matrix
conf_mat = confusion_matrix(Y_Test, y_pred_classes)
conf_mat_df = pd.DataFrame(conf_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
fig = px.imshow(conf_mat_df, text_auto= True, color_continuous_scale='Blues')
fig.update_xaxes(side='top', title_text='Predicted')
fig.update_yaxes(title_text='Actual')
fig.show()


# In[ ]:




