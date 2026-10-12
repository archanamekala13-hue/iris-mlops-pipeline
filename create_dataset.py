
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
)

data["species"] = [
    iris.target_names[i] for i in iris.target
]

data.to_csv("iris.csv", index=False)

print("Dataset created!")
print("Total rows:", len(data))
