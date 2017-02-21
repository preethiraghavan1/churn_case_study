import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import plot_partial_dependence


def load_data(filename):
    '''
    input - filename
    output - dataframe
    '''
    df = pd.read_csv(filename, parse_dates = [5,7])
    return df


def format_data(df, train = 1):
    df['one_ride'] = df.signup_date == df.last_trip_date
    df['active'] = df.last_trip_date >= '2014-06-01'
    df.active = 1. * df.active
    df['day_of_week'] = [x.weekday() for x in df.signup_date]
    df['weekend'] = (df.day_of_week == 5) | (df.day_of_week == 6)
    df.weekend = 1. * df.weekend
    df.weekday = 1. - df.weekend
    df = df.fillna(df.median())
    city_dummies = pd.get_dummies(df.city)
    dummy_names = list(city_dummies)
    df[dummy_names] = city_dummies

    df.price_proxy = df.avg_dist * df.avg_surge

    # calculate the means
    df_by_city = df.groupby('city').mean()
    df_by_city = df_by_city.reset_index()

    # put the means into original df
    df_mean = pd.merge(df, df_by_city, on = 'city')
    df['avg_dist_diff'] = df_mean['avg_dist_x'] - df_mean['avg_dist_y']
    df['avg_rating_by_driver_diff'] = df_mean['avg_rating_by_driver_x'] - df_mean['avg_rating_by_driver_y']
    df['avg_rating_of_driver_diff'] = df_mean['avg_rating_of_driver_x'] - df_mean['avg_rating_of_driver_y']
    df['avg_surge_diff'] = df_mean['avg_surge_x'] - df_mean['avg_surge_y']
    df['surge_pct_diff'] = df_mean['surge_pct_x'] - df_mean['surge_pct_y']
    #df['price_proxy_diff'] = df_mean['price_proxy_x'] - df_mean['price_proxy_y']

    phone_dummies = pd.get_dummies(df.phone)
    dummy_names = list(phone_dummies)
    df[dummy_names] = phone_dummies
    return df, df.active


def test_model(model, X, y, independent_vars):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train[independent_vars], y_train)
    y_pred = model.predict(X_test[independent_vars])
    print 'Accuracy Score: {}'.format(accuracy_score(y_pred, y_test))
    print 'Recall Score: {}'.format(recall_score(y_pred, y_test))
    print 'Precision Score: {}'.format(precision_score(y_pred, y_test))
    print 'F1 Score: {}'.format(f1_score(y_pred, y_test))
    print 'ARUC Score: {}'.format(roc_auc_score(y_pred, y_test))
    # display_coeffs(model, X, y)
    return roc_curve(y_pred, y_test)

def plot_roc(model1, model2, model3):
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(model1[0], model1[1], 'g-', label = 'model1')
    plt.plot(model2[0], model2[1], 'b-', label = 'model2')
    plt.plot(model3[0], model3[1], 'r-', label = 'model3')
    plt.legend()
    plt.show()


def display_coeffs(model, X, y):
    intercept = model.intercept_
    coefs = model.coef_
    variable_names = X.columns

    print 'Intercept: {}'.format(intercept)
    for v_num in xrange(len(variable_names)):
        print '{} Coefficient: {}'.format(variable_names[v_num], coefs[0][v_num])


def score_test_data(X_test, y_predict):
    df_predict = pd.DataFrame({'prob': y_predict[:,1], 'UserID': X_test.UserID, 'Date': X_test.YearMonth})
    df_aggregate = df_predict.groupby('UserID').mean()
    df_aggregate = df_aggregate.sort(columns = 'prob', ascending = False).reset_index()
    return df_aggregate.UserID.head(1000)


if __name__ == '__main__':
    df = load_data('Churn/churn_train.csv')
    X, y = format_data(df, train = 1)
    independent_vars = ['avg_dist_diff', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge_diff', 'trips_in_first_30_days', 'surge_pct_diff', 'iPhone', 'luxury_car_user', 'Astapor', "King's Landing"]

    # test forest and logit models
    print 'Random Forest'
    model = RandomForestClassifier(n_estimators = 100, max_depth = 3)
    forest_roc = test_model(model, X[independent_vars], y, independent_vars)
    forest_feature = model.feature_importances_
    forest_feature_scale = forest_feature/max(forest_feature)

    print '------------'
    print 'Logit Regression'
    model = LogisticRegression()
    logit_roc = test_model(model, X[independent_vars], y, independent_vars)
    logit_coefs = model.coef_[0]
    logit_coefs_scale = logit_coefs/max(abs(logit_coefs))


    print '------------'
    print 'AdaBoost'
    model = AdaBoostClassifier(n_estimators = 100, learning_rate = 1)
    adaboost_roc = test_model(model, X[independent_vars], y, independent_vars)
    boost_feature = model.feature_importances_
    boost_feature_scale = boost_feature/max(boost_feature)

    feature = pd.DataFrame([ forest_feature_scale, logit_coefs_scale, boost_feature_scale], columns = independent_vars).T
    feature.columns = ['Random Forest', 'Logit', 'AdaBoost']
    # plot_roc(forest_roc, logit_roc, adaboost_roc)

    '''
    # run the model
    model = RandomForestClassifier(n_estimators = 100, max_depth = 5)
    model.fit(X[independent_vars], y)

    df_test = load_data('test.tsv')
    X_test = format_data(df_test, train = 0)

    # NEED TO ADD BACK THE HISTORY
    X_test = add_purchase_to_test(X, X_test)

    y_predict = model.predict_proba(X_test[independent_vars])

    top_1000 = score_test_data(X_test, y_predict)
    top_1000.to_csv('top1000.csv')
    '''
