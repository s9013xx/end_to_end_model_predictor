import xgboost as xgb
import sys
# from sklearn.datasets import fetch_openml
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from timeit import default_timer as timestamp
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

# column_string = ['batchsize', 'input_image_channels', 'input_image_size', 'total_conv_acts', 'total_conv_bias', 'total_conv_filters', 'total_conv_kernelsizes', 'total_conv_paddings', 'total_conv_strides', 'total_fc_acts', 'total_fc_bias', 'total_fc_units', 'total_pool_paddings', 'total_pool_sizes', 'total_pool_strides']
# "feature" = ["batchsize", "input_image_channels", "input_image_size", "total_conv_acts",
#     "total_conv_bias", "total_conv_filters", "total_conv_kernelsizes", "total_conv_paddings",
#     "total_conv_strides", "total_fc_acts", "total_fc_bias", "total_fc_units", "total_pool_paddings",
#     "total_pool_sizes", "total_pool_strides"]

# max_depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# eta_list = [0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.001]
# num_round_list = [5, 10, 15, 20, 25, 30, 35, 40]

max_depth_list = [23]
eta_list = [0.05]
num_round_list = [30]

def main():
    column_string = []
    for i in range(207):
        column_string.append(str(i))
    print "====preparing traing/testing data====="
    titanic = pd.read_csv('data/model_one_hot/network_1_7_10000.csv', index_col=None)
    # titanic = titanic.head(5000)
    titanic = titanic.dropna()

    X = titanic[column_string]
    y = titanic['latency']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    ##### validate data #####
    # print "====preparing validate data====="
    # validate = pd.read_csv('end_to_end_model_result_one_col.csv', index_col=None)
    # validate = validate.dropna()
    # X_validate = validate[column_string]
    # y_validate = validate['latency']
    # dvalidate = xgb.DMatrix(X_validate, label=y_validate)

    print "====start training====="
    best_test_error = sys.float_info.max
    best_validate_error = sys.float_info.max
    
    for max_depth in max_depth_list:
        for eta in eta_list:
            for num_round in num_round_list:
                # Define xgboost parameters
                print(max_depth, eta, num_round)
                params = {
                    'max_depth': max_depth,
                    'eta': eta,
                    'silent': 1,
                    # 'num_class': len(label.classes_),
                    # 'objective': 'multi:softmax',
                    'objective': 'reg:linear',
                    # 'tree_method': 'exact'
                    # 'tree_method': 'gpu_hist'
                    # 'boster': 'gblinear'
                    'boster': 'gbtree'
                }
                bst = xgb.train(params, dtrain, num_round)

                ##### test #####
                preds = bst.predict(dtest)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = np.mean(np.absolute(preds-y_test))
                mape = np.mean((np.absolute(preds-y_test))/y_test) * 100
                r2 = float(1 - (np.sum(np.square(preds-y_test)) / np.sum(np.square(y_test - np.mean(y_test)))))
                if mape < best_test_error:
                    best_test_error = mape
                    print("[Test] RMSE: %f ms, MAE: %f ms, MAPE: %f %%, R2: %f" % (rmse, mae, mape, r2))

                    df_result = pd.DataFrame()
                    df_result['y_test']=y_test
                    df_result['preds']=preds
                    print(df_result)
                    # df_result.to_csv('QQQ_1.csv')

                    # preds = bst.predict(dvalidate)
                    # rmse = np.sqrt(mean_squared_error(y_validate, preds))
                    # mae = np.mean(np.absolute(preds-y_validate))
                    # mape = np.mean((np.absolute(preds-y_validate))/y_validate) * 100
                    # r2 = float(1 - (np.sum(np.square(preds-y_validate)) / np.sum(np.square(y_validate - np.mean(y_validate)))))
                    # print("[Validate] RMSE: %f ms, MAE: %f ms, MAPE: %f %%, R2: %f" % (rmse, mae, mape, r2))


                ##### validate #####
                # preds = bst.predict(dvalidate)
                # rmse = np.sqrt(mean_squared_error(y_validate, preds))
                # mae = np.mean(np.absolute(preds-y_validate))
                # mape = np.mean((np.absolute(preds-y_validate))/y_validate) * 100
                # r2 = float(1 - (np.sum(np.square(preds-y_validate)) / np.sum(np.square(y_validate - np.mean(y_validate)))))
                # if mape < best_validate_error:
                #     best_validate_error = mape
                #     print("[Validate] RMSE: %f ms, MAE: %f ms, MAPE: %f %%, R2: %f" % (rmse, mae, mape, r2))

                #     df_result = pd.DataFrame()
                #     df_result['y_validate']=y_validate
                #     df_result['preds']=preds
                #     print(df_result)
                    # df_result.to_csv('QQQ_2.csv')


if __name__ == '__main__':
    main()