import pandas as pd
import plotly.express as px
from c4der._c4der import c4der


df = pd.read_csv("../deep-hair/3D.csv", index_col=0)
df = df[df["T"] < 3320]

c4der_scan = c4der(70, -1, 90, 300, 10)
c4der_scan.fit(df[["X", "Y"]].to_numpy(), df["T"].to_numpy())
df["labels"] = c4der_scan.labels_
print(df.labels)

fig = px.scatter_3d(
    z=df["T"],
    y=df["X"],
    x=df["Y"],
    color=df["labels"],
)

#fig.show()
