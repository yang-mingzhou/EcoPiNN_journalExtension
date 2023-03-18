import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import psycopg2
import datetime
import plotly.io as pio
import time
from shapely.geometry import Polygon
import os
import gc
from os import walk
import geopandas as gpd
import math
import random
from random import shuffle
import csv
import pickle


# plot map matching results
def plotRoutes(route, name, filename):
    directory = './results'
    if not os.path.exists(directory):
        os.makedirs(directory)
    long_edge = []
    lat_edge = []
    for xs, ys in route:
        print(xs,ys)
        long_edge.append(xs)
        lat_edge.append(ys)
    fig = go.Figure(go.Scattermapbox(
        name = name,
        mode = "lines",
        lon = long_edge,
        lat = lat_edge,
        marker = {'size': 5, 'color':'green'},
        line = dict(width = 3, color = 'green')))
    # getting center for plots:
    lat_center = np.mean(lat_edge)
    long_center = np.mean(long_edge)
    zoom = 9.5
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                            'center': {'lat': lat_center,
                            'lon': long_center},
                            'zoom': zoom})
    plotly.offline.plot(fig,filename = os.path.join(directory,filename+'.html'),auto_open=True)

cnt = 0
f = r'./resultNewOct/features_percentile005'
path_list = os.listdir(f)
path_list.sort(key=lambda x:int(x[15:-4]))
for file in path_list:
    filename = f + '/' + file
    if cnt == 0:
        df_edge = pd.read_csv(filename,skiprows=[1])
    else:
        df = pd.read_csv(filename,skiprows=[1])
        df_edge = pd.concat([df_edge,df])
    cnt += 1
df_edge = df_edge.reset_index(drop = True)
# fill in Nan in speed_limit
index_emp = df_edge[df_edge['speed_limit'].isna()].index
for i in index_emp:
    road = df_edge.loc[i,'road_type']
    if road == "motorway" :
        df_edge.loc[i,'speed_limit'] = 55*1.609
    elif road == "motorway_link" :
        df_edge.loc[i,'speed_limit'] = 50*1.609
    else:
        df_edge.loc[i,'speed_limit'] = 30*1.609
# l/100km = 100 * (l/h) / (km/h)
#print(df_edge.iloc[0])
df_edge['energy_consumption_per_100km_est'] = 100 * df_edge['energy_consumption_per_hour'] / df_edge['average_speed']

df_edge['osmNodeIdUV'] = df_edge.tags.apply(lambda x: tuple(list(map(int, x[1:-1].split(", ")))[:-1]))

#dualGraphNode.index((data_row[-1][0], data_row[-1][1], 0))

file_name = "DeepLearning/dualGraphNodes.pkl"
open_file = open(file_name, "rb")
dualGraphNode = pickle.load(open_file)
open_file.close()

df_edge['osmNode'] = df_edge.osmNodeIdUV.apply(lambda x: dualGraphNode.index((x[0], x[1], 0)))

df_edge = df_edge.fillna(axis=0,method='ffill')

df_edge['segment_count'] = df_edge.groupby('trip_id')['network_id'].transform('count')

df_edge = df_edge.drop(df_edge[df_edge['segment_count']<3].index)

df_edge = df_edge.reset_index(drop = True)

percentile_1 = df_edge.energy_consumption_per_100km.quantile(0.01)

percentile_99 = df_edge.energy_consumption_per_100km.quantile(0.99)

df_edge_percent = df_edge.drop(df_edge[df_edge['energy_consumption_per_100km']>percentile_99].index).\
    drop(df_edge[df_edge['energy_consumption_per_100km']<percentile_1].index).reset_index(drop = True)

per_ele_001 = df_edge_percent.elevation_change.quantile(0.01)
per_ele_99 = df_edge_percent.elevation_change.quantile(0.99)
print(per_ele_001,per_ele_99)
df_edge_ele_percent = df_edge_percent.drop(df_edge_percent[df_edge_percent['elevation_change'] >per_ele_99].index).\
    drop(df_edge_percent[df_edge_percent['elevation_change'] < per_ele_001].index).reset_index(drop = True)

df_test = df_edge_ele_percent

counterFunc = df_test.apply(lambda x: True if abs(x['previous_orientation']) > 179 else False, axis=1)
df_test.drop(counterFunc[counterFunc == True].index,inplace=True)
df_test.reset_index(drop = True, inplace = True)
print(df_test.head())

cnt = 0
for i in range(len(df_test)):
    if i > 0 and df_test.loc[i,'trip_id'] != df_test.loc[i-1,'trip_id']:
        cnt += 1
    df_test.loc[i,'trip']  = cnt

random.seed(1234)
trip_num = len(df_test['trip_id'].unique())
k_folder_list = list(range(trip_num))
shuffle(k_folder_list)
print('num of trips', len(k_folder_list))

portionOfDataWithFuel = 0.1
outputFolderPrefix = 'recursion{}perc'.format(portionOfDataWithFuel*100)


test_list  = k_folder_list[int(0.8*len(k_folder_list)):]


k_folder_list = k_folder_list[:int(0.8*len(k_folder_list))]

for datasetenumerate in range(11,21):
    random.seed(datasetenumerate)
    shuffle(k_folder_list)
    #60-20-20
    train_list  = k_folder_list[: int(0.75*len(k_folder_list))]
    val_list  = k_folder_list[int(0.75*len(k_folder_list)):]
    train_list_3 = k_folder_list[:int(0.75*portionOfDataWithFuel*len(k_folder_list))]
    val_list_3 = k_folder_list[int(0.75*len(k_folder_list)):int((0.75+portionOfDataWithFuel*0.25)*len(k_folder_list))]
    #train_list_3 = k_folder_list[:2]
    #val_list_3 = k_folder_list[int(0.75 * len(k_folder_list)):(int(0.75 * len(k_folder_list))+2)]
    print(len(train_list), len(val_list), len(test_list), len(train_list_3), len(val_list_3))
    #80-10-10
    # train_list_1  = k_folder_list[: int(0.8*len(k_folder_list))]
    # val_list_1  = k_folder_list[int(0.8*len(k_folder_list)):int(0.9*len(k_folder_list))]
    # test_list_1  = k_folder_list[int(0.9*len(k_folder_list)):]
    # train_list_2 = k_folder_list[:int(0.008*len(k_folder_list))]
    # val_list_2 = k_folder_list[int(0.8*len(k_folder_list)):int(0.801*len(k_folder_list))]
    # print(len(train_list_1),len(val_list_1),len(test_list_1),len(train_list_2),len(val_list_2))

    df_test = df_test[['network_id', 'position',
           'road_type', 'average_speed', 'speed_limit', 'mass', 'elevation_change',
           'previous_orientation', 'length', 'energy_consumption_total','siumlatedEnergyConsumption',
           'time',  'direction_angle', 'time_stage', 'week_day',
            'lanes', 'bridge', 'endpoint_u', 'endpoint_v', 'segment_count', 'trip','osmNodeIdUV','osmNode' ]]

    df_test['segment_count'] = df_test.groupby('trip')['network_id'].transform('count')

    trip_before = -1
    position = 1
    for i in range(len(df_test)):
        if df_test.loc[i,'trip'] != trip_before:
            position = 1
            trip_before = df_test.loc[i,'trip']
        else:
            position += 1
        df_test.loc[i,'position']  = position

    d = df_test.groupby('road_type')['speed_limit'].mean()

    d.sort_values()

    dictionary = {}
    road_tp = 0
    for i in d.sort_values().index:
        dictionary[i] = road_tp
        road_tp += 1


    output_root = "DeepLearning/road_type_dictionary.csv"
    csvFile = open(output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["road type", "value"])
    for i in dictionary:
        writer.writerow([i, dictionary[i]])
    csvFile.close()
    np.save('DeepLearning/road_type_dictionary.npy', dictionary)

    endpoints_dictionary = np.load('endpoints_dictionary.npy', allow_pickle=True).item()


    output_root = "DeepLearning/endpoints_dictionary.csv"
    csvFile = open(output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["endpoint", "value"])
    for i in endpoints_dictionary:
        writer.writerow([i, endpoints_dictionary[i]])
    csvFile.close()

    df_test['road_type']=df_test['road_type'].apply(lambda x:dictionary[x])

    # for lookuptable method
    output = outputFolderPrefix + str(datasetenumerate)
    outputpath = os.path.join("lookupdata", output)
    print(outputpath)
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    df_train = df_test[df_test['trip'].apply(lambda x: x in train_list_3 or x in val_list_3)]
    df_val = df_test[df_test['trip'].apply(lambda x: x in test_list)]
    df_train.to_csv(os.path.join(outputpath,"train_data.csv"))
    df_val.to_csv(os.path.join(outputpath,"val_data.csv"))

    print('lookuptable finished')
    new_columns = [
     'average_speed',
     'speed_limit',
     'mass',
     'elevation_change',
     'previous_orientation',
     'length',
     'direction_angle',
     'network_id',
     'position',
     'road_type',
     'time_stage',
     'week_day',
     'lanes',
     'bridge',
     'endpoint_u',
     'endpoint_v',
     'energy_consumption_total',
    'siumlatedEnergyConsumption',
     'time',
     'segment_count',
     'trip',
     'osmNodeIdUV',
     'osmNode',
        ''
    ]

    df02 = df_test.reindex(columns=new_columns)

    output_root = "DeepLearning/mean_std.csv"
    csvFile = open(output_root, "w")
    writer = csv.writer(csvFile)
    writer.writerow(["attribute", "mean","std"])
    for i,val in enumerate(df02.columns):
        if i < 7:
            x_mean = df02[val].mean()
            x_std = df02[val].std()
            writer.writerow([val,x_mean,x_std])
            df02[val] = df02[val].apply(lambda x: (x - x_mean) / x_std)
        elif val == 'siumlatedEnergyConsumption' or val == 'time':
            x_mean = df02[val].mean()
            x_std = df02[val].std()
            writer.writerow([val, x_mean, x_std])
    csvFile.close()

    df_train = df02[df02['trip'].apply(lambda x: x in train_list)]

    df_val  = df02[df02['trip'].apply(lambda x: x in val_list)]
    df_t = df02[df02['trip'].apply(lambda x: x in test_list)]
    df_train.to_csv("data_dropped/20/train_data.csv")
    df_val.to_csv("data_dropped/20/val_data.csv")
    df_t.to_csv("data_dropped/20/test_data.csv")
    df_train_fuel = df02[df02['trip'].apply(lambda x: x in train_list_3)]
    df_val_fuel = df02[df02['trip'].apply(lambda x: x in val_list_3)]
    df_train_fuel.to_csv("data_dropped/20/train_data_fuel.csv")
    df_val_fuel.to_csv("data_dropped/20/val_data_fuel.csv")
    df_t.to_csv("data_dropped/20/test_data_fuel.csv")

    # df_train = df02[df02['trip'].apply(lambda x: x in train_list_1)]
    # df_val = df02[df02['trip'].apply(lambda x: x in val_list_1)]
    # df_t = df02[df02['trip'].apply(lambda x: x in test_list_1)]
    # df_train_fuel = df02[df02['trip'].apply(lambda x: x in train_list_2)]
    # df_val_fuel = df02[df02['trip'].apply(lambda x: x in val_list_2)]
    # df_train.to_csv("data_dropped/10/train_data.csv")
    # df_val.to_csv("data_dropped/10/val_data.csv")
    # df_t.to_csv("data_dropped/10/test_data.csv")
    # df_train_fuel.to_csv("data_dropped/10/train_data_fuel.csv")
    # df_val_fuel.to_csv("data_dropped/10/val_data_fuel.csv")
    # right one
    path = r'data_dropped'
    output = outputFolderPrefix+ str(datasetenumerate)
    for f,m,n in os.walk(path):
        if n:
            outputpath = os.path.join("DeepLearning/ExpDataset",output)
            print(outputpath)
            if not os.path.exists(outputpath):
                os.mkdir(outputpath)
            outputpath = os.path.join(outputpath,f[-2:])
            print(outputpath)
            if not os.path.exists(outputpath):
                os.mkdir(outputpath)
            #print(f,m,n)
            for file in n:
                f_n = os.path.join(f,file)
                df_train = pd.read_csv(f_n, header=0)
                #print(df_train.describe())
                df_train = df_train.fillna(axis=0,method='ffill')
                #print(df_train.describe())
                # df_train['energy_consumption_total'] = df_train['energy_consumption_total'].apply(lambda x: 100*x)
                df_train['data'] = df_train.apply(lambda x: [x['speed_limit'],x['mass'],x['elevation_change'],x['previous_orientation'],x['length'],x['direction_angle']], axis = 1)
                # smoothed data
                df_train['label'] = df_train.apply(lambda x: [x["siumlatedEnergyConsumption"],x["time"]], axis = 1)
                # act data
                #df_train['label'] = df_train.apply(lambda x: [x["energy_consumption_total"],x["time"]], axis = 1)
                trip_before = -1
                position = 1
                for i in range(len(df_train)):
                    if df_train.loc[i,'trip'] != trip_before:
                        position = 1
                        trip_before = df_train.loc[i,'trip']
                    else:
                        position += 1
                    df_train.loc[i,'position_new'] = position
                df_train['trip'] = df_train['trip'].apply(lambda x: int(x))
                df_train = df_train[['data','label','network_id','segment_count',"position_new","road_type","time_stage", "week_day", "lanes", "bridge", "endpoint_u", "endpoint_v","trip",'osmNode']]
                print("finish")
                df_train.to_csv(os.path.join(outputpath,file),header=False, index = False)


