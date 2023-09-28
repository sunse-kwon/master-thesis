import numpy as np
import pandas as pd


# define manual train-test split function
def train_test_split(df, test_step_size=672):
    Train = pd.DataFrame()
    Test = pd.DataFrame()
    for key in df.carpark_number.value_counts().keys():
        inner_df = df[df.carpark_number == key]
        train = inner_df.iloc[0:-test_step_size,:]
        test = inner_df.iloc[-test_step_size:,:]
        Train = pd.concat([Train,train],ignore_index=True)
        Test = pd.concat([Test,test],ignore_index=True)
    return Train, Test


# define window generator for ML models
def window_generator(df, window_size=1,label_col_no=0):
    availability, others = np.split(df,[3],axis=1)
    l =[]
    for i in others.columns:
        l.append(others[i].unique())
    exo = np.hstack(l)
    
    availability = availability.transpose()
    D = availability.values
    rows, columns = D.shape
    exo = np.reshape(exo, (1,-1))
    
    x = []
    y = []
    for col in range(columns -window_size):
        exo = exo
        d = D[:,col:col+window_size]
        d = np.reshape(d, (1,-1))
        x.append(np.hstack([exo,d]))
        y.append(D[:,col+window_size][label_col_no]) 
    
    x = np.vstack(x)
    y = np.array(y)
    return x,y


#define window generator for DL models
def window_generator_DL(df, window_size=1, label_col_no=0):
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np) -window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        x.append(row)
        label = df_as_np[i+window_size, label_col_no]
        y.append(label)
    return np.array(x), np.array(y)


#define window generator for Seq2seq models
def window_generator_Seq2seq(df, x_width=1, y_width=1, label_col_no=0):
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np)+1 -x_width -y_width):
        row = [r for r in df_as_np[i:i+x_width]]
        x.append(row)
        label = df_as_np[i+x_width:i+x_width+y_width, label_col_no]
        y.append(label)
    return np.array(x), np.array(y)


# splitting x,y using window generator
def X_Y_split(df, window_size=5,label_col_no=4):
  #split x,y per parking lot
  X = []
  Y = []
  for key in df.carpark_number.value_counts().keys():
    inner_df = df[df.carpark_number == key]
    x,y = window_generator(inner_df, window_size=window_size, label_col_no=label_col_no)
    X.append(x)
    Y.append(y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y


#splitting x,y for deep learning models
def X_Y_split_DL(df, window_size=5, label_col_no=0):
  #split x,y for each parking lot
  X = []
  Y = []
  for key in df.carpark_number.value_counts().keys():
          inner_df = df[df.carpark_number == key]
          x,y = window_generator_DL(inner_df, window_size=window_size, label_col_no=label_col_no)
          X.append(x)
          Y.append(y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y


#splitting x,y for Seq2seq models
def X_Y_split_Seq2seq(df, x_width=5, y_width=5, label_col_no=0):
  #split x,y for each parking lot
  X = []
  Y = []
  for key in sorted(df.carpark_number.value_counts().keys()):
          inner_df = df[df.carpark_number == key]
          x,y = window_generator_Seq2seq(inner_df, x_width=x_width,y_width=y_width, label_col_no=label_col_no)
          X.append(x)
          Y.append(y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y


# getting last timestep data for recursive strategy for ML models
def last_x_y_generator(df, window_size=5,label_col_no=4):
  #split x,y per each parking lot
  X = []
  Y = []
  for key in sorted(df.carpark_number.value_counts().keys()):
    inner_df = df[df.carpark_number == key]
    x,y = window_generator(inner_df, window_size=window_size, label_col_no=label_col_no)
    last_x = x[-1:,:]
    last_y = y[-1:]
    X.append(last_x)
    Y.append(last_y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y


# getting last timestep data for recursive strategy for DL models
def last_x_y_generator_DL(df, window_size=5, label_col_no=0):
  #split x,y for each parking lot
  X = []
  Y = []
  for key in sorted(df.carpark_number.value_counts().keys()):
          inner_df = df[df.carpark_number == key]
          x,y = window_generator_DL(inner_df, window_size=window_size, label_col_no=label_col_no)
          last_x = x[-1:,:,:]
          last_y = y[-1:]
          X.append(last_x)
          Y.append(last_y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y



def last_x_y_generator_Seq2seq(df, x_width=5, y_width=5, label_col_no=0):
  #split x,y for each parking lot
  X = []
  Y = []
  for key in sorted(df.carpark_number.value_counts().keys()):
          inner_df = df[df.carpark_number == key]
          x = window_generator_Seq2seq(inner_df, x_width=x_width,y_width=y_width, label_col_no=label_col_no)
          last_x = x[-1:,:,:]
          last_y = y[-1:]
          X.append(last_x)
          Y.append(last_y)

  X = np.vstack(X)
  Y = np.concatenate(Y)
  return X, Y



# scaler for subsetting dataset
def scaler(Train,Test):
    for i in Train.columns:
        scaler = MinMaxScaler()
        s_train = scaler.fit_transform(Train[i].values.reshape((-1,1)))
        s_train = np.reshape(s_train,(len(s_train)))
        Train[i] = s_train   
        s_test = scaler.transform(Test[i].values.reshape((-1,1)))
        s_test = np.reshape(s_test,(len(s_test)))
        Test[i] = s_test
    return Train,Test


# function to get next prediction value using single feature (lots_available)
def insert_end(Xin, new_input, timestep):
    #print ('Before: \n', Xin , new_input )
    for i in range(10,10+timestep-1):
        Xin[:,i] = Xin[:,i+1]
    Xin[:,10+timestep-1] = new_input
    #print ('After :\n', Xin)
    return Xin


def insert_end_DL(Xin,new_input, timestep):
    #print ('Before: \n', Xin , new_input )
    for i in range(timestep-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,timestep-1,0:1] = new_input
    #print ('After :\n', Xin)
    return Xin



def insert_end_Seq2seq(Xin,new_input, timestep):
    #print ('Before: \n', Xin , new_input )
    for i in range(timestep-4):
        Xin[:,i,:] = Xin[:,i+4,:]
    Xin[:,timestep-4:,0:1] = new_input
    #print ('After :\n', Xin)
    return Xin



# function to get next prediction using multiple features (lots_available, day_of_week, hour_of_day)
def insert_end_multi(Xin, new_input, timestep):
    #print ('Before: \n', Xin , new_input )
    for i in range(timestep-1):
        Xin[:,i+10] = Xin[:,i+10+1]
        Xin[:,i+26] = Xin[:,i+26+1]
        Xin[:,i+42] = Xin[:,i+42+1]
    Xin[:,10+timestep-1] = new_input
    #print ('After :\n', Xin)
    return Xin
