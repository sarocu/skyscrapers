import json
import csv
import pandas

from models.pca import Principle_Components


class Index:
    def __init__(self):
        with open("./sp.json", "r") as configs:
            self.config = json.load(configs)

        self.data = {}
        self.load_data()

    def load_data(self):
        for dataset in self.config["data"]["dataframe"]:
            if dataset["type"] == "csv":
                self.data[dataset["name"]] = pandas.read_csv(dataset["path"])[
                    [*dataset["vars"]]
                ]
                if dataset["datetime"]:
                    self.data[dataset["name"]][
                        dataset["datetime"]
                    ] = pandas.to_datetime(
                        self.data[dataset["name"]][dataset["datetime"]]
                    )
            elif dataset["type"] == "json":
                self.data[dataset["name"]] = json.load(dataset["path"])
            else:
                continue


project = Index()

"""
    Your code starts here:
"""
