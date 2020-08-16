import json
import csv
import pandas
import numpy
import logging
import logging.handlers

from models.mlpr import MLPR

handler = logging.handlers.RotatingFileHandler(
    filename="./output/project.log",
    maxBytes=1000000
)
handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
log = logging.getLogger("project.error")
log.setLevel("INFO")
log.addHandler(handler)

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
                    print(dataset["name"])
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
large_school = project.data["features"][project.data["features"].building_id == 7]
log.info(large_school)

"""
Training Data:
- consumption (measures via the meter reading) is the response, here we look at only one meter type
- weather variables are the predictors
- we combine the data into a single dataframe to ensure the timeseries line up
- we then clean the data by removing NaNs and replacing with the previous value
"""
school_consumption = project.data["consumption"][project.data["consumption"].building_id == int(large_school.building_id)]
school_consumption.set_index('timestamp', inplace=True)
log.info(school_consumption[:5])
school_consumption = school_consumption[school_consumption.meter == 1].fillna(method='pad')

school_weather = project.data["weather"][project.data["weather"].site_id == int(large_school.site_id)].fillna(method='pad')
school_weather.set_index('timestamp', inplace=True)
log.info(school_weather[:5])
training_data = school_weather.join(school_consumption, on='timestamp', how='left')
log.info(training_data[:5])

# clean the combined DF:
training_data = training_data.replace([numpy.inf, -numpy.inf], numpy.nan)
training_data = training_data.fillna(method='pad')[['air_temperature', 'dew_temperature', 'meter_reading', 'meter']]
training_data = training_data[training_data['meter_reading'].notna()]
# save for reuse:
training_data.to_csv("./data/training.csv")
model = MLPR(
    predictor=training_data[['air_temperature', 'dew_temperature']],
    response=training_data.meter_reading
)