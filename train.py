#Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(
        random_state=1,
        max_depth=5,        # limit depth
        min_samples_leaf=4  # minimum samples per leaf
    ))
])

#Random Forest Model
from sklearn.ensemble import RandomForestRegressor
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,    # number of trees
        random_state=1,
        n_jobs=-1            # use all CPU cores
    ))
])

# Split data (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

#Fitting DT
dt_pipeline.fit(X_train, y_train)

#Fitting RF
rf_pipeline.fit(X_train, y_train)
