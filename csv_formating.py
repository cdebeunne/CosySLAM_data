from contextlib import suppress
import numpy as np
import pandas as pd
from joblib import load
import pinocchio as pin

def error_to_covariance(error):
    cov = np.zeros((6,6))
    for i in range(6):
        val = error[i]*error[i]
        if val < 1e-6:
            val = 1e-6
        cov[i,i] = val
    return cov

if __name__ == '__main__':
    error_model = load('error_model.pkl')

    alias = 'demo2'
    data_path = f'{alias}/'

    # processing of the data frame
    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]
    cosy_score = df_cosypose['detection_score'].values
    cosy_object = df_cosypose['object_name'].values

    covariance_list = []
    dt = 1/30
    for i in range(len(cosy_score)):
        b_M_c = c_M_b_cosy[i].inverse()

        if cosy_object[i] == "obj_000023":
            # error model parameters
            r = np.linalg.norm(b_M_c.translation)
            phi = np.arcsin(b_M_c.translation[2]/r)
            theta = np.arctan(b_M_c.translation[0]/b_M_c.translation[1])

            detection_score = cosy_score[i]
            if i==0:
                speed = 0
                b_M_c_prev = b_M_c
            else:
                speed = np.linalg.norm(
                    (b_M_c_prev.translation-b_M_c.inverse().translation)/dt)
                b_M_c_prev = b_M_c
            error = error_model.predict([[r, theta, phi, detection_score, speed]])[0]
            cov = error_to_covariance(error)
            cov = np.array2string(cov, separator=' ',suppress_small=True, max_line_width=10000).replace('\n',' ').replace('[',' ').replace(']',' ')
            covariance_list.append(cov)
        else:
            error = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
            cov = error_to_covariance(error)
            cov = np.array2string(cov, separator=' ',suppress_small=True, max_line_width=10000).replace('\n',' ').replace('[',' ').replace(']',' ')
            covariance_list.append(cov)
        
    
    np.set_printoptions(suppress=True)
    df_cosypose['covariance'] = covariance_list
    df_cosypose['timestamp'] = df_cosypose['timestamp'] - df_cosypose['timestamp'][0]
    # format pose
    str_pose = [np.array2string(pose, separator=' ',suppress_small=True, max_line_width=10000).replace('\n','').replace('[','').replace(']','') for pose in df_cosypose['pose']]
    df_cosypose['pose'] = str_pose
    
    df_cosypose.to_csv(f'input_{alias}.csv',  float_format='%f', encoding='utf-8')

