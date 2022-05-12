import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime, date


def main():

    # DATA FROM BEFORE 2019-07-08
    df1 = pd.read_csv('data/health_workout.csv')
    #print(df1)
    df1['dates'] = df1['startDate'].apply(lambda d: datetime.strptime(d[:10], '%Y-%m-%d'))
    df1['totalDistance'] = df1['totalDistance'].apply(lambda x: float(x[:4]))
    df1 = df1.groupby('dates').agg(distance=pd.NamedAgg(column='totalDistance', aggfunc=sum))
    df1 = df1.reindex(pd.date_range('05-17-2017', '12-13-2021'), fill_value=0)
    date_limit = datetime.strptime('2019-07-08', '%Y-%m-%d')
    df1 = df1[df1.index < date_limit]
    #print(df1)

    # DATA FROM 2019-07-08 ONWARD
    df = pd.read_csv('data/strava_data.csv', usecols=['name', 'start_time', 'distance', 'moving_time', 'elapsed_time', 'workout_type'])
    df['workout_type'].fillna(0, inplace=True)
    df['dates'] = df['start_time'].apply(lambda d: datetime.strptime(d[:10], '%Y-%m-%d').date())
    #print(df['dates'])
    df = df.groupby('dates').agg(distance=pd.NamedAgg(column='distance', aggfunc=sum))
    df = df.reindex(pd.date_range('07-08-2019', '12-13-2021'), fill_value=0)


    final = pd.concat([df1, df], axis=0)
    final['distance_sma'] = final['distance'].rolling(42).mean() * 7
    final.fillna(0, inplace=True)




    # import race data
    df = pd.read_csv('race_data.csv')
    df = df[df['tag'] == 'real']
    df.index = df['start_time'].apply(lambda d: datetime.strptime(str(d), '%Y-%m-%d'))
    print(df)

    ann_df = pd.concat([final[final.index.isin(df.index)]['distance_sma'], df[['distance', 'xc']]], axis=1)
    #df = df.drop(df.index[df['distance_sma'].isna()])

    print(df)


    X_train = ann_df[['distance_sma', 'distance', 'xc']].values.reshape(-1, 3)
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = df['time_seconds'].values.reshape(-1,1)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = pd.read_csv('fake_data.csv').dropna().values.reshape(-1, 3)
    X_test = np.asarray(X_test).astype(np.float32)
    print(X_train.shape, y_train.shape)

    model = Sequential()
    model.add(Dense(units=1000, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, batch_size=2, epochs=200, verbose=1)

    predictions = model.predict(X_test)
    print(predictions)

    fake_data = pd.read_csv('fake_data.csv').dropna()
    fake_xc = fake_data[fake_data['xc'] == True]['race_distance']
    fake_track = fake_data[fake_data['xc'] == False]['race_distance']
    predictions_xc = predictions[fake_data.index[fake_data['xc'] == True]]
    predictions_track = predictions[fake_data.index[fake_data['xc'] != True]]

    plt.scatter(fake_xc, predictions_xc, color='blue')
    plt.scatter(fake_track, predictions_track, color='orange')
    plt.show()


    #plt.plot(final)
    #plt.show()

    #print(df)
    #print(df[df['name'] == 'cooldown'])

    # read in race data

    # plt.scatter(df[df['tag'] == 'real']['distance'], df[df['tag'] == 'real']['time_seconds'])
    # plt.plot(df[df['tag'] == 'min']['distance'], df[df['tag'] == 'min']['time_seconds'], color='green')
    # plt.plot(df[df['tag'] == 'max']['distance'], df[df['tag'] == 'max']['time_seconds'], color='red')
    # plt.xlim([0,2])
    # plt.ylim([0,300])
    # plt.show()



if __name__ == '__main__':
    main()

