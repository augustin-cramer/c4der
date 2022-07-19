from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
from c4der._c4der import c4der


def standardize_cluster_format(converter) -> dict:
    i = 0
    for k, v in converter.items():
        if v and k != -1:
            converter[k] = i
            i += 1
        else:
            converter[k] = -1

    return converter


if __name__ == "__main__":

    df = pd.read_csv("test_set/20220627.csv", index_col=0)
    df['rectangularity']  = df.lengths / df.widths
    df['timestamps']  = pd.to_datetime(df['timestamps'] )
    df['time_seconds'] = df['timestamps'].apply(lambda x : pd.Timedelta(x- df.loc[0,'timestamps']).total_seconds() )
    ########### c4der Time ###########
    c4der_scan = c4der(
        50, -1, 70, 360, 10, spatial_weight_on_x=0.2, non_spatial_epss=0.1
    )

    c4der_scan.fit(
        df[["x", "y"]].to_numpy(),
        df['time_seconds'].to_numpy(),
        #X_non_spatial=df["rectangularity"].to_numpy(),
    )
    df["labels"] = c4der_scan.labels_

    ###################################

    converter = standardize_cluster_format((df.groupby("labels").count().x > 20))
    # Maybe look at cluster time_span instead of nb of points
    
    df["labels"] = df["labels"].apply(lambda x: converter[x])

    #df = df[df.labels != -1]  # Get rid of the noise


    fig = px.scatter_3d(
        z=df["timestamps"],
        y=df["x"],
        x=df["y"],
        color=df["labels"],
    )

    fig.show()

    print(f"Nombre de clients détectés :{sum((df.labels.unique()!=-1).astype(int))}")
    for cluster in df.labels.unique():
        if cluster>=0:
            start = min(df[df.labels == cluster]["timestamps"])
            end = max(df[df.labels == cluster]["timestamps"])
            print(f'{start.strftime("%H:%M")} -->  {end.strftime("%H:%M")}')
