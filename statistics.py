import numpy as np
from numpy.lib.polynomial import poly
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import utils.posemath as pm
from utils.wrapper import ErrorWrapper

if __name__ == '__main__':
    
    aliases = ['switch1', 'switch3', 'switch4', 'switch5']
    aliases = ['legrand1', 'legrand2', 'legrand4']
    aliases = ['campbell1', 'campbell2', 'campbell3']
    error_wrapper = ErrorWrapper('obj_000004', 'data/')
    df = error_wrapper.create_df(aliases)

    # formatting the data 
    X = df[['r','theta', 'phi', 'detection_score']]
    Y = df[['translation_err_0', 'translation_err_1', 'translation_err_2',
        'angular_err_0', 'angular_err_1', 'angular_err_2']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # let's set and train a polynomial regression pipeline 
    degree = 2
    model = LinearRegression()
    poly = PolynomialFeatures(degree)
    polyreg=make_pipeline(poly,model)
    polyreg.fit(X,Y)

    # An example with typical values
    r = 0.35
    theta = 0.03
    phi = 0.42
    s = 0.999995
    pred = polyreg.predict([[r,theta,phi,s]])
    print(pred[0,:3])
    print(pm.log3_to_euler(pred[0,3:]))

    # for degree 2 and 4 variables: 
    # [1 x1 x2 x3 x4 x1^2 x1*x2 x1*x3 x1*x4 x2^2 x2*x3 x2*x4 x3^2 x4*x3 x4^2]

    print("------- MODEL EVALUATION ---------")

    print("model score: "+str(polyreg.score(X, Y)))
    print("Model coeficients: ")
    np.set_printoptions(suppress=True)
    print(np.array2string(model.coef_, separator=', '))
    print(model.intercept_)

    # compute RMSE and R2 score 
    y_test_pred = polyreg.predict(X_test)
    rmse_trans = np.sqrt(mean_squared_error(y_test[['translation_err_0','translation_err_1','translation_err_2']],
                y_test_pred[:,:3]))
    rmse_ang = np.sqrt(mean_squared_error(y_test[['angular_err_0','angular_err_1','angular_err_2']],
                y_test_pred[:,3:]))
    r2 = r2_score(y_test,y_test_pred)
    print("RMSE on the test samples for translation error:")
    print(rmse_trans)
    print("RMSE on the test samples for angular error:")
    print(rmse_ang*180/3.14)
    print("R2 on the test samples: ")
    print(r2)

    y_train_pred = polyreg.predict(X_train)
    rmse_trans = np.sqrt(mean_squared_error(y_train[['translation_err_0','translation_err_1','translation_err_2']],
                y_train_pred[:,:3]))
    rmse_ang = np.sqrt(mean_squared_error(y_train[['angular_err_0','angular_err_1','angular_err_2']],
                y_train_pred[:,3:]))
    r2 = r2_score(y_train,y_train_pred)
    print("RMSE on the training samples for translation error:")
    print(rmse_trans)
    print("RMSE on the training samples for angular error:")
    print(rmse_ang*180/3.14)
    print("R2 on the training samples: ")
    print(r2)


    pred = polyreg.predict(X)
    plt.plot(X['r'], pred[:,0], '.', label='model')
    plt.plot(X['r'], Y['translation_err_0'],'.' ,label='groundtruth')
    plt.legend()
    plt.show()


