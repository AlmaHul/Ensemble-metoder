from sklearn.ensemble import RandomForestRegressor

def train_rf(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
