from sklearn.neural_network import MLPRegressor


class MLPR:
    def __init__(self, predictor, response, random_state=1, max_iterations=500):
        self.predictor = predictor
        self.response = response
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.model = MLPRegressor(
            random_state=random_state, 
            max_iter=max_iterations
            )
        self.model.fit(predictor, response)

    def predict(self, test):
        return self.model.predict(test)

    def run_test(self, predictor_test, response_test):
        return self.model.score(predictor_test, response_test)
