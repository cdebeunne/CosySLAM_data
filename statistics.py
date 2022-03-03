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
import seaborn as sns
sns.set()
import utils.posemath as pm
from utils.wrapper import ErrorWrapper

"""
The script to produce the polynomial error models

Enter the aliases of the trajectories you need to fit your model 
and specify the object label for the errorWrapper in the main
"""

if __name__ == '__main__':
    
    aliases = ['legrand1', 'legrand2', 'legrand4']
    aliases = ['campbell1', 'campbell2', 'campbell3']
    aliases = ['newLegrand1', 'newLegrand2', 'newLegrand3', 'newLegrand4', 'newLegrand5']
    aliases = ['switch1', 'switch4', 'switch5']
    error_wrapper = ErrorWrapper('obj_000026', 'data/')
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
    theta = np.mean(X['theta'])
    phi = np.mean(X['phi'])
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
    plt.plot(X['phi'], pred[:,0], '.', label='model')
    plt.plot(X['phi'], Y['translation_err_0'],'.' ,label='groundtruth')
    plt.legend()
    plt.show()

    s_list = np.linspace(0.98,1,100)
    X_det = [[0.4,theta,phi,s] for s in s_list]
    pred_det = polyreg.predict(X_det)
    trans_err = [np.linalg.norm(np.array([x,y,z])) for x,y,z in zip(pred_det[:,0],pred_det[:,1], pred_det[:,2])]
    plt.plot(s_list, trans_err)
    plt.show()

    r_list = np.linspace(0.3,0.8,100)
    X_det = [[r,0.0,0.25,0.999993] for r in r_list]
    pred_det = polyreg.predict(X_det)
    trans_err = [np.linalg.norm(np.array([x,y,z])) for x,y,z in zip(pred_det[:,0],pred_det[:,1], pred_det[:,2])]
    np.savez('plot_data.npz', X_det=X_det, trans_err=trans_err)
    plt.plot(r_list, trans_err)
    plt.show()



