import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def get_data(file_name, pollution):
    data = pd.read_csv(file_name, sep=';')
    pollution = pollution.upper()
    if pollution == 'PM25':
        data = data.drop('PM10', axis=1)
    elif pollution == 'PM10':
        data = data.drop('PM25', axis=1)

    dates = data['date'].to_list()[3:]
    data = data.drop('date', axis=1)
    data = data[[pollution, 'TEMP', 'HUMIDITY', 'WIND SPEED']]

    x = []
    measurements = data.values.tolist()
    for i in range(len(measurements) - 3):
        x.append(measurements[i+3][1:] + measurements[i+2] + measurements[i+1] + measurements[i])
    y = data[pollution].tolist()[3:]

    return np.array(x), np.array(y), np.array(dates)


def split_data(data, trainPercentage):
    splitPoint = int(trainPercentage/100*len(data))
    trainData = data[:splitPoint]
    testData = data[splitPoint:]

    return trainData, testData


def normalize(data, ifReverse, maximum, minimum, lowLimit, highLimit):
    if ifReverse:
        data = (data - lowLimit)/(highLimit - lowLimit)*(maximum - minimum) + minimum
    else:
        data = (highLimit - lowLimit)*(data - minimum)/(maximum - minimum) + lowLimit

    return data


def activation_function(x, function):
    if function == 'sigmoid':
        return 1/(1 + np.exp(-x))
    elif function == 'tanh':
        return 2/(1 + np.exp(-2*x)) - 1


def perceptron(inputArray, weights):
    return activation_function(np.dot(inputArray, weights[1:]) + weights[0], function)


def hidden_layer(inputArray, weights):
    return [perceptron(inputArray, weights[i, :inputArray.shape[0]+1]) for i in range(hidden)]


def net(inputArray, weights):
    hiddenLayerOutput = hidden_layer(inputArray, weights)
    outputLayerOutput = [perceptron(hiddenLayerOutput, weights[-1, :hidden+1])]
    sumOutput = hiddenLayerOutput + outputLayerOutput

    return sumOutput


def learning_sigmoid(trainInputs, trainOutputs, weights, gamma):
    for i in range(trainInputs.shape[0]):
        sumOutput = net(trainInputs[i], weights)
        y = sumOutput[-1]

        #outputLayer
        weights[-1, 0] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * y * (1 - y) * 1
        for s in range(1, hidden + 1):
            weights[-1, s] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * y * (1 - y) * sumOutput[s - 1]

        #inputLayer
        for h in range(hidden):
            weights[h, 0] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * y * (1 - y) * sumOutput[h] * (1 - sumOutput[h]) * weights[-1, h]
            for p in range(1, inputs):
                weights[h, p] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * y * (1 - y) * sumOutput[h] * (1 - sumOutput[h]) * weights[-1, h] * trainInputs[i, p-1]

    return weights


def learning_tanh(trainInputs, trainOutputs, weights, gamma):
    for i in range(trainInputs.shape[0]):
        sumOutput = net(trainInputs[i], weights)
        y = sumOutput[-1]

        #outputLayer
        weights[-1, 0] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * (1 - y**2) * 1
        for s in range(1, hidden + 1):
            weights[-1, s] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * (1 - y**2) * sumOutput[s - 1]

        #inputLayer
        for h in range(hidden):
            weights[h, 0] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * (1 - y**2) * (1 - sumOutput[h]**2) * weights[-1, h]
            for p in range(1, inputs):
                weights[h, p] -= gamma * 2 * (trainOutputs[i] - y) * (-1) * (1 - y**2) * (1 - sumOutput[h]**2) * weights[-1, h] * trainInputs[i, p-1]

    return weights


def compute_model(x, y, hidden_layer, epochLimit, datasetSplitRatio, function, gamma):
    output = 1
    hidden = hidden_layer
    min, max = -25, 200

    if function == 'sigmoid':
        low = 0
    elif function == 'tanh':
        low = -1

    x = normalize(x, 0, max, min, low, 1)
    y = normalize(y, 0, max, min, low, 1)
    x_train, x_test = split_data(x, datasetSplitRatio)
    y_train, y_test = split_data(y, datasetSplitRatio)
    inputs = x_train.shape[1]
    weights = 2 * np.random.random_sample((hidden + 1, inputs + 1 + hidden + 1 + output)) - 1

    print("Rozpoczęto proces uczenia się sieci...")
    epoch = 0

    while epoch < epochLimit:
        if function == 'sigmoid':
            learning_sigmoid(x_train, y_train, weights, gamma)
        elif function == 'tanh':
            learning_tanh(x_train, y_train, weights, gamma)

        '''If you want to see results of neural network after every epoch of learning process uncomment below section'''

        # x = normalize(x, 1, max, min, low, 1)
        # y = normalize(y, 1, max, min, low, 1)
        # errorTrain, errorTest = test_model(x, y, datasetSplitRatio, weights)
        # x = normalize(x, 0, max, min, low, 1)
        # y = normalize(y, 0, max, min, low, 1)
        # print(f"Epoch {epoch},\tTrain error: {round(errorTrain, 2)},\tTest error: {round(errorTest, 2)}")

        epoch += 1
    print('Proces uczenia się sieci zakończony.')

    return weights


def test_model(x, y, datasetSplitRatio, weightsf):

    if function == 'sigmoid':
        low = 0
    elif function == 'tanh':
        low = -1

    x_train, x_test = split_data(x, datasetSplitRatio)
    y_train, y_test = split_data(y, datasetSplitRatio)
    x_train = normalize(x_train, 0, max, min, low, 1)
    x_test = normalize(x_test, 0, max, min, low, 1)

    y_pred = []
    summarizedErrorTrain = 0

    for i in range(x_train.shape[0]):
        results = net(x_train[i], weightsf)
        y_pred.append(normalize(results[-1], 1, max, min, low, 1))
        error = abs(y_pred[i] - y_train[i])
        summarizedErrorTrain += error
    print("\nŚredni bezwzględny błąd na ciągu treningowym: ", summarizedErrorTrain/y_train.shape[0])

    y_pred_n = []
    summarizedErrorTest = 0

    for i in range(x_test.shape[0]):
        results = net(x_test[i], weightsf)
        y_pred_n.append(normalize(results[-1], 1, max, min, low, 1))
        error = abs(y_pred_n[i] - y_test[i])
        summarizedErrorTest += error
    print("Średni bezwzględny błąd na ciągu testowym: ", summarizedErrorTest/y_test.shape[0])

    return [summarizedErrorTrain/y_train.shape[0], summarizedErrorTest/y_test.shape[0]]


def get_air_quality_data(dates):
    http = "https://api.waqi.info/feed/warsaw/?token=dd136eafa5af3050d948849dacc8c348d1e4b33f"
    response = requests.get(http)
    pm10data = response.json()['data']['forecast']['daily']['pm10']
    pm25data = response.json()['data']['forecast']['daily']['pm25']
    pm10DataFrame = pd.json_normalize(pm10data)
    pm25DataFrame = pd.json_normalize(pm25data)
    combinedDataFrame = pd.DataFrame({'Date': dates, 'pm10': pm10DataFrame['avg'].head(4).values.tolist(), 'pm25': pm25DataFrame['avg'].head(4).values.tolist()})

    return combinedDataFrame


def get_weather_data():
    https = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m,relativehumidity_2m,windspeed_10m&past_days=2"
    response = requests.get(https)
    data = response.json()
    dates = data['hourly']['time']
    for i in range(len(dates)):
        dates[i] = datetime.strptime(dates[i][:10], '%Y-%m-%d')
    df = pd.DataFrame(dates, columns=['DATE'])
    df['TEMPERATURE'] = data['hourly']['temperature_2m']
    df['HUMIDITY'] = data['hourly']['relativehumidity_2m']
    df['WIND SPEED'] = data['hourly']['windspeed_10m']
    df = df.set_index('DATE')
    df_avg = df.resample('D').mean().round(1).head(4)
    array = df_avg.values.tolist()
    array = sum(array, [])

    return array


def get_dates():
    twoDaysAgo = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    yestarday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.today().strftime("%Y-%m-%d")
    tomorrow = (datetime.today() - timedelta(days=-1)).strftime("%Y-%m-%d")
    dates = [twoDaysAgo, yestarday, today, tomorrow]

    return dates


def normColors(values, pollution):
    colorNames = ['green', 'limegreen', 'gold', 'darkorange', 'red', 'maroon']
    colorIDs = ['color', 'color', 'color']
    pollutionID = 0
    values = [float(value) for value in values]

    if pollution == 'PM25':
        pollutionID = 1

    norms = [[20, 50, 80, 110, 150], [13, 35, 55, 75, 110]]

    for i in range(len(values)):
        if values[i] <= norms[pollutionID][0]:
            colorIDs[i] = 0
        elif values[i] <= norms[pollutionID][1]:
            colorIDs[i] = 1
        elif values[i] <= norms[pollutionID][2]:
            colorIDs[i] = 2
        elif values[i] <= norms[pollutionID][3]:
            colorIDs[i] = 3
        elif values[i] <= norms[pollutionID][4]:
            colorIDs[i] = 4
        else:
            colorIDs[i] = 5

    colors = [colorNames[colorIDs[0]], colorNames[colorIDs[1]], colorNames[colorIDs[2]]]

    return colorIDs, colors


def getPollutionAdvice(levelPM10, levelPM25):
    messages = ['', '', '']

    for i in range(len(levelPM10)):
        level = np.max(np.array(levelPM10[i], levelPM25[i]))

        if level == 0:
            messages[i] = ('Brak zagrożenia dla zdrowia, warunki sprzyjające wszelkim aktywnościom na wolnym powietrzu')
        elif level == 1:
            messages[i] = ('Niskie ryzyko zagrożenia dla zdrowia. Można jednak przebywać na wolnym powietrzu i wykonywać dowolne aktywności')
        elif level == 2:
            messages[i] = ('Możliwe zagrożenie dla zdrowia w szczególnych przypadkach - dla osób chorych, osób starszych,'
                            ' kobiet w ciąży oraz małych dzieci. Warunki umiarkowane do aktywności na wolnym powietrzu.')
        elif level == 3:
            messages[i] = ('Zagrożenie dla zdrowia, szczególnie dla chorych, starszych, kobiet w ciąży oraz małych'
                            ' dzieci. Możliwe negatywne skutki zdrowotne. Rozważyć ograniczenie aktywności na wolnym powietrzu')
        elif level == 4:
            messages[i] = ('Jakość powietrza jest zła. Osoby chore, starsze, kobiety w ciąży oraz małe dzieci powinny unikać'
                            ' przebywania na wolnym powietrzu. Pozostali powinni ograniczyć do minimum wszelką aktywność na wolnym powietrzu')
        elif level == 5:
            messages[i] = ('Negatywny wpływ na zdrowie. Osoby chore, starsze, kobiety w ciąży oraz małe dzieci powinny '
                            'bezwzględnie unikać przebywania na wolnym powietrzu. Pozostała populacja powinna '
                            'ograniczyć przebywanie na wolnym powietrzu do niezbędnego minimum. Wszelkie aktywności '
                            'fizyczne na zewnątrz są odradzane.')

    return messages


def weatherRoot():
    weatherroot = tk.Tk()
    weatherroot.title('Dane meteorologiczne')
    weatherroot.geometry("1270x180")

    for i in range(3):
        dayFrame = tk.LabelFrame(weatherroot, text=dayLabels[i], width=430, height=500)
        dayFrame.grid(column=i, row=0)
        yesterdayWeather = tk.LabelFrame(dayFrame, text=weatherLabels[i], width=410, height=140)
        yesterdayWeather.grid(column=0, row=0)
        yesterdayWeatherValues = tk.Label(yesterdayWeather,
                                      text=f'Temperatura:\t{weather[i][0]} [°C]\nWilgotność:\t{weather[i][1]} %\nPrędkość wiatru:\t{weather[i][2]} [km/h]',
                                      font=('Arial', 28), justify=tk.LEFT)
        yesterdayWeatherValues.grid(column=0, row=0)
        yesterdayWeather.grid_propagate(0)


if __name__ == '__main__':
    pollutions = ['pm10', 'pm25']
    max, min = 200, -25
    hidden, output, gamma, epochLimit, trainPercentage, function = 3, 1, 0.007, 1500, 70, 'tanh'
    pollutionMeasurements = []
    predictions = []

    for pollution in pollutions:
        try:
            frame = pd.read_excel('Wagi_' + pollution + '.xlsx')
            frame = frame.drop(frame.columns[0], axis=1)
            weights = np.array(frame.values.tolist())
            x, y, dates = get_data('Dane Warszawa.csv', pollution)
            print('Zanieczyszczenie ' + pollution.upper() + ":")
            test_model(x, y, 70, weights)
        except FileNotFoundError:
            x, y, dates = get_data('Dane Warszawa.csv', pollution)
            inputs = x[0].shape[0]
            weights = compute_model(x, y, hidden, epochLimit, trainPercentage, function, gamma)
            weightsDataFrame = pd.DataFrame(np.array(weights))
            weightsDataFrame.to_excel('Wagi_' + pollution + '.xlsx')

        dates = get_dates()
        airQualityData = get_air_quality_data(dates)
        pollutionData = airQualityData[pollution].values.tolist()
        weatherData = get_weather_data()
        x = weatherData[0:3] + [pollutionData[0]] + weatherData[3:6] + [pollutionData[1]] + weatherData[6:9] + [pollutionData[2]] + weatherData[9:12]
        x_n = normalize(np.array(x), 0, max, min, -1, 1)
        netResponse = net(x_n, weights)

        #Predykcja na jutro
        prediction = normalize(netResponse[-1], 1, max, min, -1, 1)
        predictions.append(np.round(prediction, 2))
        pollutionMeasurements.append([pollutionData[1], pollutionData[2]])
        print('\n' + pollution.upper(), 'Predykcja na jutro: ', prediction, '\n')

    result = tk.Tk()
    result.title('System Predykcji Jakości Powietrza')
    result.geometry("1270x700")
    style = ttk.Style(result)
    style.theme_use('classic')
    dayLabels = ['Wczoraj', 'Dzisiaj', 'Jutro']
    measurementsPM25 = [pollutionMeasurements[1][0], pollutionMeasurements[1][1], '-']
    measurementsPM10 = [pollutionMeasurements[0][0], pollutionMeasurements[0][1], '-']

    historicalPredictions = pd.read_excel('Predykcje.xlsx', index_col=0)
    historicalPredictions['Date'] = historicalPredictions['Date'].astype("string")
    todayPredictions = pd.DataFrame(np.array([[((datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')), predictions[0], predictions[1]]]), columns=['Date', 'PM10', 'PM25'])

    connectedPredictions = pd.concat([historicalPredictions, todayPredictions])
    connectedPredictions = connectedPredictions.drop_duplicates(['Date']).reset_index(drop=True)
    connectedPredictions.to_excel("Predykcje.xlsx")

    weather = [weatherData[3:6], weatherData[6:9], weatherData[9:12]]
    weatherLabels = ['Pomiary meteorologiczne', 'Pomiary meteorologiczne', 'Prognoza metorologiczna']

    predicionsPM25 = [round(float(connectedPredictions.iloc[-3]['PM25'])), round(float(connectedPredictions.iloc[-2]['PM25'])),
                      round(float(connectedPredictions.iloc[-1]['PM25']))]
    predicionsPM10 = [round(float(connectedPredictions.iloc[-3]['PM10'])), round(float(connectedPredictions.iloc[-2]['PM10'])),
                      round(float(connectedPredictions.iloc[-1]['PM10']))]

    levelPM25, colorsPM25 = normColors(predicionsPM25, 'PM25')
    levelPM10, colorsPM10 = normColors(predicionsPM10, 'PM10')
    advices = getPollutionAdvice(np.array(levelPM25), np.array(levelPM10))

    if connectedPredictions.iloc[-2]['Date'] != dates[-2]:
        predicionsPM25[-2], predicionsPM10[-2], colorsPM10[-2], colorsPM25[-2], advices[-2] = '-', '-', 'black', 'black', 'Predykcja dla tego dnia nie została wykonana.'
    if connectedPredictions.iloc[-3]['Date'] != dates[-3]:
        predicionsPM25[-3], predicionsPM10[-3], colorsPM10[-3], colorsPM25[-3], advices[-3] = '-', '-', 'black', 'black', 'Predykcja dla tego dnia nie została wykonana.'

    pm10bar = Image.open("PM10bar.png")
    pm10bar = pm10bar.resize((400, 60), Image.ANTIALIAS)
    pm10bar = ImageTk.PhotoImage(pm10bar)

    pm25bar = Image.open("PM25bar.png")
    pm25bar = pm25bar.resize((400, 60), Image.ANTIALIAS)
    pm25bar = ImageTk.PhotoImage(pm25bar)


    for i in range(3):
        dayFrame = tk.LabelFrame(result, text=dayLabels[i], width=430, height=500)
        dayFrame.grid(column=i, row=0)
        prediction = tk.LabelFrame(dayFrame, text='Predykcja zanieczyszczeń', width=430, height=210)
        prediction.grid(column=0, row=0)
        yesterdayPredictionPM10 = tk.LabelFrame(prediction, text='PM10 Predykcja', width=205, height=160)
        yesterdayPredictionPM10.grid(column=0, row=0)
        yesterdayPredictionPM10.grid_propagate(0)
        yesterdayPredictionPM10Value = ttk.Label(yesterdayPredictionPM10, text=predicionsPM10[i], font=('Arial', 116),
                                                 width=3, anchor=tk.CENTER, background='#fff', foreground=colorsPM10[i])
        yesterdayPredictionPM10Value.grid(column=0, row=0, sticky='s')

        yesterdayPredictionPM25 = tk.LabelFrame(prediction, text='PM25 Predykcja', width=205, height=160)
        yesterdayPredictionPM25.grid(column=1, row=0)
        yesterdayPredictionPM25.grid_propagate(0)
        yesterdayPredictionPM25Value = ttk.Label(yesterdayPredictionPM25, text=predicionsPM25[i], font=('Arial', 116),
                                                width=3, anchor=tk.CENTER, background='#fff', foreground=colorsPM25[i])
        yesterdayPredictionPM25Value.grid(column=0, row=0)

        if predicionsPM10[i] == '-':
            pointer = "▼" + 110 * ' '
        else:
            pointer = int(np.round((predicionsPM10[i] / 2), 0)) * ' ' + "▼" + (108 - int(np.round((predicionsPM10[i] / 2), 0))) * ' '

        barPM10frame = tk.LabelFrame(dayFrame, width=500, height=60, text='PM10 Skala')
        barPM10frame.grid(column=0, row=1)
        txt = tk.Label(barPM10frame, text=pointer)
        txt.grid(column=0, row=0, sticky='S')
        barPM10 = tk.Label(barPM10frame, image=pm10bar)
        barPM10.grid(column=0, row=1)

        if predicionsPM25[i] == '-':
            pointer = "▼" + 110 * ' '
        else:
            pointer = int(np.round((predicionsPM25[i] / 2), 0)) * ' ' + "▼" + (104 - int(np.round((predicionsPM25[i] / 2), 0))) * ' '

        barPM25frame = tk.LabelFrame(dayFrame, width=500, height=60, text='PM25 Skala')
        barPM25frame.grid(column=0, row=2)
        txt = tk.Label(barPM25frame, text=pointer)
        txt.grid(column=0, row=0)
        barPM25 = tk.Label(barPM25frame, image=pm25bar)
        barPM25.grid(column=0, row=1)

        advicePM25 = tk.LabelFrame(dayFrame, text='Interpretacja predykcji', width=410, height=130)
        advicePM25.grid(column=0, row=3)
        adviceTextPM25 = tk.Label(advicePM25, text=advices[i], wraplength=390, justify=tk.CENTER, anchor=tk.CENTER)
        adviceTextPM25.place(x=0, y=0)

        yesterdayMeasurement = tk.LabelFrame(dayFrame, text='Pomiar Zanieczyszczeń', width=410, height=100)
        yesterdayMeasurement.grid(column=0, row=4)
        yesterdayMeasurementValues = tk.Label(yesterdayMeasurement,
                                              text=f'PM10: {measurementsPM10[i]}\nPM25: {measurementsPM25[i]}',
                                              font=('Arial', 30), justify=tk.LEFT)
        yesterdayMeasurementValues.grid(column=0, row=0)
        yesterdayMeasurement.grid_propagate(0)

        weatherButton = tk.Button(result, text='Pokaż dane meteorologiczne', command=weatherRoot, height=2, width=45)
        weatherButton.grid(column=1, row=1)

    result.mainloop()
