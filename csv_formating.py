import numpy as np
import pandas as pd
import pinocchio as pin

if __name__ == '__main__':

    alias = 'demo2'
    data_path = 'data/'

    # processing of the data frame
    df_cosypose = pd.read_pickle(data_path+f'results_{alias}_ts.pkl')
    df_cosypose = df_cosypose.loc[df_cosypose['pose'].notnull()]
    c_M_b_cosy = [pin.SE3(T) for T in df_cosypose['pose']]
    cosy_score = df_cosypose['detection_score'].values
    cosy_object = df_cosypose['object_name'].values
    
    np.set_printoptions(suppress=True)
    df_cosypose['timestamp'] = df_cosypose['timestamp'] - df_cosypose['timestamp'][0]
    # format pose
    str_pose = ([np.array2string(pose, separator=' ',suppress_small=True,
     max_line_width=10000).replace('\n','').replace('[','').replace(']','') for pose in df_cosypose['pose']])
    df_cosypose['pose'] = str_pose
    
    df_cosypose.to_csv(f'input_{alias}.csv',  float_format='%f', encoding='utf-8')

