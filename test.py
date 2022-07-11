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

    df = pd.read_csv("test_set/3D_500_area.csv", index_col=0)
    df.Area = np.sqrt(df.Area)
    start = datetime(year=2022, month=7, day=2, hour=9, minute=1)
    df.loc[:, "T"] = df["T"].apply(lambda x: start + timedelta(seconds=x))

    ########### c4der Time ###########
    c4der_scan = c4der(
        50, -1, 70, 240, 10, spatial_weight_on_x=0.2, non_spatial_epss=0.1
    )

    c4der_scan.fit(
        df[["X", "Y"]].to_numpy(),
        df["T"].to_numpy(),
        X_non_spatial=df["Area"].to_numpy(),
    )
    df["labels"] = c4der_scan.labels_

    ###################################

    converter = standardize_cluster_format((df.groupby("labels").count().X > 20))
    # Maybe look at cluster time_span instead of nb of points
    
    df["labels"] = df["labels"].apply(lambda x: converter[x])

    df = df[df.labels != -1]  # Get rid of the noise

    fig = px.scatter_3d(
        z=df["T"],
        y=df["X"],
        x=df["Y"],
        color=df["labels"],
    )

    fig.show()

    print(f"Nombre de clients détectés :{len(df.labels.unique())}")
    for cluster in df.labels.unique():
        start = min(df[df.labels == cluster]["T"])
        end = max(df[df.labels == cluster]["T"])
        print(f'{start.strftime("%H:%M")} -->  {end.strftime("%H:%M")}')
