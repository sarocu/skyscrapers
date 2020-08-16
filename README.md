# ASHRAE Building Energy Data
Based on the data provided by the ASHRAE Kaggle competition: https://www.kaggle.com/c/ashrae-energy-prediction

## Made with Superplus
Superplus is a CLI made to make statistical learning and data based projects easier and more fun, automating the trivial stuff so you can focus on what your data means. 

```python
sp new project --name buildings
sp new csv -p ~/train.csv
...
sp new model
```

After setting up any data transformations and importing your models, run scripts via Superplus:
```python
sp run lint
sp run start
```

And add a basic line chart to plot the predictions:
```python
sp new chart
```
