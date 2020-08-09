import numpy
from sklearn.decomposition import PCA


class Principle_Components:
    def __init__(self, data, max_components, solver="auto"):
        self.data = data
        self.solver
        self.max_components = max_components
        self.model = PCA(max_components, solver)
        self.model.fix(data)

    def refit(self, max_components=self.max_components, solver=self.solver):
        self.solver = solver
        self.max_components = max_components
        self.model = PCA(max_components, solver)
        self.model.fit(self.data)
        return self

    def results(self):
        return {
            "explained_variance": self.model.explained_variance_ratio_,
            "singular_values": self.model.singular_values_,
            "score": self.model.score,
        }
