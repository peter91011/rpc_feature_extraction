from __future__ import absolute_import

from pyspark.sql import functions as func
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext, SparkSession
from pyspark.sql.functions import col, struct, to_json, udf, expr
from pyspark.sql.window import Window
from pyspark.sql.types import *

import argparse
from multiprocessing import Pool
import sys, os, json, shutil, glob, warnings, gc, random, collections, csv, re, logging, time, pickle
from pathlib import Path
from datetime import datetime
from google.cloud import storage

import pandas as pd
import numpy as np
import lightgbm as lgb
import gcsfs

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from utils.utils import *

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=UserWarning)

def read_one_partition(filepath):
    """
    load one partition and reduce memory usage.
    """
    df = pd.read_csv(filepath, header=None, sep='\t', quoting=csv.QUOTE_NONE)
    df, _ = reduce_mem_usage(df)
    return df

def read_data_parallel(filepath, schema, cores=16):
    all_partitions = extract_gcs_files(filepath)
    df = []
    with Pool(processes=cores) as pool:
        df_list = pool.map(read_one_partition, all_partitions)
        pool.close()
        pool.join()
        df = pd.concat(df_list, ignore_index=True)
        df.columns = schema
    logging.info(f"Data shape: {df.shape}")
    return df

def wrmse_old(y, y_pred, weights):
    error = np.power(y - y_pred, 2)*np.log(weights + np.e)
    return np.sqrt(np.sum(error)/np.sum(np.log(weights + np.e)))
def wmae_old(y, y_pred, weights):
    error = np.abs(y - y_pred)*np.log(weights + np.e)
    return np.sum(error)/np.sum(np.log(weights + np.e))

def wmae(y, y_pred, weights):
    error = np.abs(y - y_pred)*weights
    return np.sum(error)/np.sum(weights)
def wrmse(y, y_pred, weights):
    error = np.power(y - y_pred, 2)*weights
    return np.sqrt(np.sum(error)/np.sum(weights))

def train(df_trn, df_val, feature_cols, target_col, lgb_params, args, model_name, weight_col=None):
    # build dataset
    if weight_col:
        dtrain = lgb.Dataset(df_trn[feature_cols],
                             df_trn[target_col].values,
                             weight=df_trn[weight_col].values,
                             #weight=np.log(df_trn[weight_col].values + np.e),
                             free_raw_data=False,
                             silent=False,
                             )
        dvalid = lgb.Dataset(df_val[feature_cols],
                             df_val[target_col].values,
                             weight=df_val[weight_col].values,
                             #weight=np.log(df_val[weight_col].values + np.e),
                             free_raw_data=False,
                             silent=False,
                             )
    else:
        dtrain = lgb.Dataset(df_trn[feature_cols],
                             df_trn[target_col].values,
                             free_raw_data=False,
                             silent=False,
                             )
        dvalid = lgb.Dataset(df_val[feature_cols],
                             df_val[target_col].values,
                             free_raw_data=False,
                             silent=False,
                             )
    # train model
    model = lgb.train(lgb_params,
                      dtrain,
                      valid_sets=[dtrain, dvalid],
                      verbose_eval=100)

    local_file = f'{args.train_path_local}{model_name}.txt'
    model.save_model(local_file, num_iteration=model.best_iteration)

    gcp_file = f'{args.train_path_gcs}{model_name}.txt'
    upload_local_to_cloud_storage(local_file, gcp_file)


def eval(df, feature_cols, target_col, args, model_name, weight_col=None):
    try:
        download_from_cloud_storage(f'{args.train_path_local}{model_name}.txt', f'{args.train_path_gcs}{model_name}.txt')
        model = lgb.Booster(model_file=(f'{args.train_path_local}/{model_name}.txt'))
    except:
        raise RuntimeError(f'{model_name} does not exist...')

    y_pred = model.predict(df[feature_cols])
    y_pred = np.array([max(0,x) if np.isnan(x)==False else x for x in y_pred])
    if weight_col:
        WRMSE = wrmse(df[target_col].values, y_pred, df[weight_col].values)
        WMAE = wmae(df[target_col].values, y_pred, df[weight_col].values)
    else:
        raise RuntimeError('No weight column. ')
    logging.info(f'WRMSE: {WRMSE: .5f}')
    logging.info(f'WMAE: {WMAE: .5f}')
    return y_pred

def score(df, feature_cols, args, model_name):
    # download trained model from gcp
    local_file = f'{args.train_path_local}{model_name}.txt'
    download_from_cloud_storage(local_file, f'{args.train_path_gcs}/{model_name}.txt')

    # load model and score
    try:
        model = lgb.Booster(model_file=local_file)
    except:
        raise RuntimeError(f'{local_file} does not exist...')
    y_pred = model.predict(df[feature_cols])
    y_pred = np.array([max(0,x) if np.isnan(x)==False else x for x in y_pred])
    return y_pred

def assign_confidence(df):
    all_cols = df.columns
    pattern1 = np.any(df[[col for col in all_cols if re.match(r"(^sem_.*_w1$)", col)]].values > 0, axis=1)
    pattern2 = np.any(df[[col for col in all_cols if re.match(r"(^sem_.*_w2$)", col)]].values > 0, axis=1)
    pattern3 = np.any(df[[col for col in all_cols if re.match(r"(^sem_.*_w3$)", col)]].values > 0, axis=1)
    df['confidence'] = 4
    df['confidence'].loc[pattern3] = 3
    df['confidence'].loc[pattern2] = 2
    df['confidence'].loc[pattern1] = 1
    return df.confidence.tolist()


def run_train(args):
    # make sure model type is correct
    if args.model_type not in ['convrt', 'ordersize', 'all', 'rpc']:
        raise RuntimeError(f"Expected model type to be ['convrt', 'ordersize', 'all', 'rpc'], but get '{args.model_type}' instead.")
    if args.target not in ['seller', 'brand', 'all']:
        raise RuntimeError(f"Expected target to be ['seller', 'brand', 'all'], but get '{args.target}' instead.")

    # lgb paramters
    lgb_params = {
        'n_estimators': 4000,
        'metric': 'rmse',
        'learning_rate': 0.02,
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'subsample': 0.9,
        'bagging_freq': 10,
        'colsample_bytree': 0.33,
        'verbose': -1,
        'num_leaves': 63,
        'max_depth': 10,
        'seed': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 200,
    }

    logging.info("Prepare training data...")

    # load data from gcp bucket
    config = load_json(args.data_config)
    adid_name = config["hash_id"]["adid"]
    is_mobile_name = config["hash_id"]["is_mobile"]
    source_id_name = config["hash_id"]["source_id"]
#     df_eval1 = pd.read_csv('sample_data_full_feat/df_feat_all_coop_item_'+'2022-12-19'+'.csv')
#     df_eval1 = df_eval1[(df_eval1['sem_orders_brand_l'].isnull()==False)&(df_eval1['sem_orders_seller_l'].isnull()==False)]
#     df_eval2 = pd.read_csv('sample_data_full_feat/df_feat_all_coop_item_'+'2022-12-26'+'.csv')
#     df_eval2 = df_eval2[(df_eval2['sem_orders_brand_l'].isnull()==False)&(df_eval2['sem_orders_seller_l'].isnull()==False)]
#     df_eval3 = pd.read_csv('sample_data_full_feat/df_feat_all_coop_item_'+'2023-01-02'+'.csv')
#     df_eval3 = df_eval3[(df_eval3['sem_orders_brand_l'].isnull()==False)&(df_eval3['sem_orders_seller_l'].isnull()==False)]
#     df_eval4 = pd.read_csv('sample_data_full_feat/df_feat_all_coop_item_'+'2023-01-09'+'.csv')
#     df_eval4 = df_eval4[(df_eval4['sem_orders_brand_l'].isnull()==False)&(df_eval4['sem_orders_seller_l'].isnull()==False)]
#     df_eval5 = pd.read_csv('sample_data_full_feat/df_feat_all_coop_item_'+'2023-01-16'+'.csv')
#     df_eval5 = df_eval5[(df_eval5['sem_orders_brand_l'].isnull()==False)&(df_eval5['sem_orders_seller_l'].isnull()==False)]
#     df_eval = pd.concat([df_eval1,df_eval2,df_eval3,df_eval4,df_eval5])
#     del df_eval1
#     del df_eval2
#     del df_eval3
#     del df_eval4
    filename = args.path_gcs+'sample_data_full_feat/df_feat_all_coop_item_'+args.ref_date+'.csv'
    df_eval = pd.read_csv(filename)
    # sort by the least number of null values per row and select training size
    #df_eval = df_eval.iloc[df_eval.isnull().sum(1).sort_values(ascending=True).index]
    #df_eval = df_eval.sort_values('sem_clicks_l',ascending=False)
    #df_eval = df_eval[(df_eval['sem_orders_brand_l'].isnull()==False)&(df_eval['sem_orders_seller_l'].isnull()==False)]
    df_eval = df_eval.sort_values('sem_clicks_w1',ascending=False)
    # fill na with 0 for all columns including targets
    for i in [k for k in df_eval.columns if '_l' in k]:
        df_eval[i] = df_eval[i].fillna(0)
    df_eval = df_eval.head(500000)
    df = df_eval.copy()
    logging.info("training data size: "+str(len(df)))
    logging.info("eval data size: "+str(len(df_eval)))
    logging.info("training data prepared...")
    
    # prepare train features
    features = config["features"]
    # clean features (all should be numerics)
    features_set = ['uber_curr_item_price_w1', 'uber_avg_overall_rating_w1'] #hard coded for hot fix
    df = convert_numeric(df, features_set)

    # do train-val split
    df['fold'] = 0
    kf = KFold(5, shuffle=True, random_state=42)
    for i, (_, val_idx) in enumerate(kf.split(df)):
        df['fold'].iloc[val_idx] = i

    # filters
    convrt_trn_brand_filter = (df[source_id_name]==2) & (df.sem_convrt_brand_l>=0) & (df.sem_convrt_brand_l<=0.3)
    convrt_val_brand_filter = (df[source_id_name]==2) & (df.sem_convrt_brand_l>=0) & (df.sem_convrt_brand_l<=0.3)
#    ordersize_trn_filter = (df[source_id_name]==2) & (df.sem_orders_l>=3) & (df.sem_ordersize_l>=0) & (df.sem_ordersize_l<=500)
    ordersize_trn_brand_filter = (df[source_id_name]==2) & (df.sem_ordersize_brand_l>=0) & (df.sem_ordersize_brand_l<=500)
    ordersize_val_brand_filter = (df[source_id_name]==2) & (df.sem_ordersize_brand_l>=0) & (df.sem_ordersize_brand_l<=500)
    rpc_trn_brand_filter = (df[source_id_name]==2) & (df.sem_ordersize_brand_l>=0) & (df.sem_ordersize_brand_l<=500)
    rpc_val_brand_filter = (df[source_id_name]==2) & (df.sem_ordersize_brand_l>=0) & (df.sem_ordersize_brand_l<=500)
    
    convrt_trn_seller_filter = (df[source_id_name]==2) & (df.sem_convrt_seller_l>=0) & (df.sem_convrt_seller_l<=0.3)
    convrt_val_seller_filter = (df[source_id_name]==2) & (df.sem_convrt_seller_l>=0) & (df.sem_convrt_seller_l<=0.3)
#    ordersize_trn_filter = (df[source_id_name]==2) & (df.sem_orders_l>=3) & (df.sem_ordersize_l>=0) & (df.sem_ordersize_l<=500)
    ordersize_trn_seller_filter = (df[source_id_name]==2) & (df.sem_ordersize_seller_l>=0) & (df.sem_ordersize_seller_l<=500)
    ordersize_val_seller_filter = (df[source_id_name]==2) & (df.sem_ordersize_seller_l>=0) & (df.sem_ordersize_seller_l<=500)
    rpc_trn_seller_filter = (df[source_id_name]==2) & (df.sem_ordersize_seller_l>=0) & (df.sem_ordersize_seller_l<=500)
    rpc_val_seller_filter = (df[source_id_name]==2) & (df.sem_ordersize_seller_l>=0) & (df.sem_ordersize_seller_l<=500)
    filter_dic = {'convrt':{'brand':[convrt_trn_brand_filter, convrt_val_brand_filter], 'seller':[convrt_trn_seller_filter,convrt_val_seller_filter]},\
                  'ordersize':{'brand':[ordersize_trn_brand_filter, ordersize_val_brand_filter], 'seller':[ordersize_trn_seller_filter,ordersize_val_seller_filter]},\
                  'rpc':{'brand':[rpc_trn_brand_filter, rpc_val_brand_filter], 'seller':[rpc_trn_seller_filter,rpc_val_seller_filter]}}

    # do train & eval
    # prepare dataframe to store OOF results
    if args.target in ['brand', 'all']:
        if args.model_type == 'all':
            df_eval['pred_convrt_brand'] = np.nan
            df_eval['pred_ordersize_brand'] = np.nan
            df_eval['pred_rpc_brand'] = np.nan
            df_eval['pred_rpc_new_brand'] = np.nan
        elif args.model_type == 'convrt':
            df_eval['pred_convrt_brand'] = np.nan
        elif args.model_type == 'ordersize':
            df_eval['pred_ordersize_brand'] = np.nan
        elif args.model_type == 'rpc':
            df_eval['pred_rpc_new_brand'] = np.nan
    if args.target in ['seller', 'all']:
        if args.model_type == 'all':
            df_eval['pred_convrt_seller'] = np.nan
            df_eval['pred_ordersize_seller'] = np.nan
            df_eval['pred_rpc_seller'] = np.nan
            df_eval['pred_rpc_new_seller'] = np.nan
        elif args.model_type == 'convrt':
            df_eval['pred_convrt_seller'] = np.nan
        elif args.model_type == 'ordersize':
            df_eval['pred_ordersize_seller'] = np.nan
        elif args.model_type == 'rpc':
            df_eval['pred_rpc_new_seller'] = np.nan
            
    targets = ['seller', 'brand'] if args.target=='all' else [args.target]
    for tar in targets:
        for fold in range(5):
            if args.model_type in ['convrt', 'all']:
                # convrt model
                logging.info('-'*20)
                logging.info(f'Train convrt model fold-{fold}'+'_'+tar+'...')
                model_features = features['convrt'][tar]
                target_col = "sem_convrt_"+tar+"_l"
                model_name = f'convrt_f{fold}'+'_'+tar
                trn_mask = filter_dic['convrt'][tar][0] & (df.fold != fold)
                df_trn = df[trn_mask]
                val_mask = filter_dic['convrt'][tar][1] & (df.fold == fold)
                df_val = df[val_mask]
                train(df_trn, df_val, model_features, target_col, lgb_params, args, model_name, weight_col='sem_clicks_l')

                # Evaluate
                logging.info(f'Eval convrt fold-{fold}'+'_'+tar+'...')
                y_pred = eval(df_val, model_features, target_col, args, model_name, weight_col='sem_clicks_l')
                df_eval['pred_convrt_'+tar].loc[val_mask] = y_pred

            if args.model_type in ['ordersize', 'all']:
                # order_size model
                logging.info('-'*20)
                logging.info(f'Train ordersize model fold-{fold}'+'_'+tar+'...')
                model_features = features['ordersize'][tar]
                target_col = "sem_ordersize_"+tar+"_l"
                model_name = f'ordersize_f{fold}'+'_'+tar
                trn_mask = filter_dic['ordersize'][tar][0] & (df.fold != fold)
                df_trn = df[trn_mask]
                val_mask = filter_dic['ordersize'][tar][1] & (df.fold == fold)
                df_val = df[val_mask]
                train(df_trn, df_val, model_features, target_col, lgb_params, args, model_name, weight_col='sem_orders_'+tar+'_l')

                # Evaluate
                logging.info(f'Eval ordersize fold-{fold}'+'_'+tar+'...')
                y_pred = eval(df_val, model_features, target_col, args, model_name, weight_col='sem_orders_'+tar+'_l')
                df_eval['pred_ordersize_'+tar].loc[val_mask] = y_pred
                
            if args.model_type in ['rpc', 'all']:
                # directly predict rpc model
                logging.info('-'*20)
                logging.info(f'Train rpc model fold-{fold}'+'_'+tar+'...')
                model_features = features['ordersize'][tar]
                target_col = "sem_rpc_"+tar+"_l"
                model_name = f'rpc_f{fold}'+'_'+tar
                trn_mask = filter_dic['rpc'][tar][0] & (df.fold != fold)
                df_trn = df[trn_mask]
                val_mask = filter_dic['rpc'][tar][1] & (df.fold == fold)
                df_val = df[val_mask]
                train(df_trn, df_val, model_features, target_col, lgb_params, args, model_name, weight_col='sem_clicks_l')

                # Evaluate
                logging.info(f'Eval rpc fold-{fold}'+'_'+tar+'...')
                y_pred = eval(df_val, model_features, target_col, args, model_name, weight_col='sem_clicks_l')
                df_eval['pred_rpc_new_'+tar].loc[val_mask] = y_pred
            if args.mode == "train":
                break
    
    # report and store oof results

    for tar in targets:
        logging.info('\n\n\nEvaluating results for '+tar)
        use_cols = []
        
        # get wmae for using avg rpc of the last 2 weeks based on different department to predict
        if args.baseline_model:
            logging.info('\nevaluating results based on avg for the last 2 weeks for each department for '+tar)
            if args.model_type == 'all':
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_rpc_'+tar+'_w1':'mean','sem_rpc_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_rpc_dept_'+tar] = (df_eval_group['sem_rpc_'+tar+'_w1']+df_eval_group['sem_rpc_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_rpc_dept_'+tar]], on = 'uber_dept_w1', how='left')
                
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_convrt_'+tar+'_w1':'mean','sem_convrt_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_convrt_dept_'+tar] = (df_eval_group['sem_convrt_'+tar+'_w1']+df_eval_group['sem_convrt_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_convrt_dept_'+tar]], on = 'uber_dept_w1', how='left')
                
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_ordersize_'+tar+'_w1':'mean','sem_ordersize_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_ordersize_dept_'+tar] = (df_eval_group['sem_ordersize_'+tar+'_w1']+df_eval_group['sem_ordersize_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_ordersize_dept_'+tar]], on = 'uber_dept_w1', how='left')

                target_cols = ['sem_convrt_'+tar+'_l', 'sem_ordersize_'+tar+'_l', 'sem_rpc_'+tar+'_l']
                pred_cols = ['pred_convrt_dept_'+tar, 'pred_ordersize_dept_'+tar, 'pred_rpc_dept_'+tar]
                weight_cols = ['sem_clicks_l', 'sem_orders_'+tar+'_l', 'sem_clicks_l']
                use_cols = use_cols + target_cols + pred_cols + weight_cols
                for target_col, pred_col, weight_col in zip(target_cols, pred_cols, weight_cols):
                    filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                    WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WRMSE: {WRMSE: .5f}")
                    logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WMAE: {WMAE: .5f}")
            elif args.model_type == 'convrt':
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_convrt_'+tar+'_w1':'mean','sem_convrt_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_convrt_dept_'+tar] = (df_eval_group['sem_convrt_'+tar+'_w1']+df_eval_group['sem_convrt_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_convrt_dept_'+tar]], on = 'uber_dept_w1', how='left')
                target_col, pred_col, weight_col = 'sem_convrt_'+tar+'_l', 'pred_convrt_dept_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WMAE: {WMAE: .5f}")  
            elif args.model_type == 'ordersize':
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_ordersize_'+tar+'_w1':'mean','sem_ordersize_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_ordersize_dept_'+tar] = (df_eval_group['sem_ordersize_'+tar+'_w1']+df_eval_group['sem_ordersize_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_ordersize_dept_'+tar]], on = 'uber_dept_w1', how='left')
                target_col, pred_col, weight_col = 'sem_ordersize_'+tar+'_l', 'pred_ordersize_dept_'+tar, 'sem_orders_'+tar+'_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WMAE: {WMAE: .5f}")  
            elif args.model_type == 'rpc':
                df_eval_group = df_eval.groupby('uber_dept_w1').agg({'sem_rpc_'+tar+'_w1':'mean','sem_rpc_'+tar+'_w2':'mean'}).reset_index()
                df_eval_group['pred_rpc_dept_'+tar] = (df_eval_group['sem_rpc_'+tar+'_w1']+df_eval_group['sem_rpc_'+tar+'_w2'])/2
                df_eval = df_eval.merge(df_eval_group[['uber_dept_w1','pred_rpc_dept_'+tar]], on = 'uber_dept_w1', how='left')
                target_col, pred_col, weight_col = 'sem_rpc_'+tar+'_l', 'pred_rpc_dept_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[1]} oof based on avg for the last 2 weeks for each dept WMAE: {WMAE: .5f}")  
        
        # get wmae for using avg rpc of the last 2 weeks to predict
        if args.baseline_model:
            logging.info('\nevaluating results based on avg for the last 2 weeks for '+tar)
            if args.model_type == 'all':
                df_eval['pred_avg_2w_convrt_'+tar] = (df_eval['sem_convrt_'+tar+'_w1']+df_eval['sem_convrt_'+tar+'_w2'])/2
                df_eval['pred_avg_2w_ordersize_'+tar] = (df_eval['sem_ordersize_'+tar+'_w1']+df_eval['sem_ordersize_'+tar+'_w2'])/2
                df_eval['pred_avg_2w_rpc_'+tar] = (df_eval['sem_rpc_'+tar+'_w1']+df_eval['sem_rpc_'+tar+'_w2'])/2
                target_cols = ['sem_convrt_'+tar+'_l', 'sem_ordersize_'+tar+'_l', 'sem_rpc_'+tar+'_l']
                pred_cols = ['pred_avg_2w_convrt_'+tar, 'pred_avg_2w_ordersize_'+tar, 'pred_avg_2w_rpc_'+tar]
                weight_cols = ['sem_clicks_l', 'sem_orders_'+tar+'_l', 'sem_clicks_l']
                use_cols = use_cols + target_cols + pred_cols + weight_cols
                for target_col, pred_col, weight_col in zip(target_cols, pred_cols, weight_cols):
                    filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                    WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WRMSE: {WRMSE: .5f}")
                    logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WMAE: {WMAE: .5f}")
            elif args.model_type == 'convrt':
                df_eval['pred_avg_2w_convrt_'+tar] = (df_eval['sem_convrt_'+tar+'_w1']+df_eval['sem_convrt_'+tar+'_w2'])/2
                target_col, pred_col, weight_col = 'sem_convrt_'+tar+'_l', 'pred_avg_2w_convrt_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WMAE: {WMAE: .5f}")  
            elif args.model_type == 'ordersize':
                df_eval['pred_avg_2w_ordersize_'+tar] = (df_eval['sem_ordersize_'+tar+'_w1']+df_eval['sem_ordersize_'+tar+'_w2'])/2
                target_col, pred_col, weight_col = 'sem_ordersize_'+tar+'_l', 'pred_avg_2w_ordersize_'+tar, 'sem_orders_'+tar+'_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WMAE: {WMAE: .5f}")  
            elif args.model_type == 'rpc':
                df_eval['pred_avg_2w_rpc_'+tar] = (df_eval['sem_rpc_'+tar+'_w1']+df_eval['sem_rpc_'+tar+'_w2'])/2
                target_col, pred_col, weight_col = 'sem_rpc_'+tar+'_l', 'pred_avg_2w_rpc_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[3]} oof based on avg for the last 2 weeks WMAE: {WMAE: .5f}")  
   
        # get wmae for using avg rpc to predict
        if args.baseline_model:
            logging.info('\nevaluating results based on avg for '+tar)
            if args.model_type == 'all':
                df_eval['pred_avg_convrt_'+tar] = np.mean(df_eval['sem_convrt_'+tar+'_l'])
                df_eval['pred_avg_ordersize_'+tar] = np.mean(df_eval['sem_ordersize_'+tar+'_l'])
                df_eval['pred_avg_rpc_'+tar] = np.mean(df_eval['sem_rpc_'+tar+'_l'])
                target_cols = ['sem_convrt_'+tar+'_l', 'sem_ordersize_'+tar+'_l', 'sem_rpc_'+tar+'_l']
                pred_cols = ['pred_avg_convrt_'+tar, 'pred_avg_ordersize_'+tar, 'pred_avg_rpc_'+tar]
                weight_cols = ['sem_clicks_l', 'sem_orders_'+tar+'_l', 'sem_clicks_l']
                use_cols = use_cols + target_cols + pred_cols + weight_cols
                for target_col, pred_col, weight_col in zip(target_cols, pred_cols, weight_cols):
                    filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                    WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                    logging.info(f"{pred_col.split('_')[2]} oof based on avg WRMSE: {WRMSE: .5f}")
                    logging.info(f"{pred_col.split('_')[2]} oof based on avg WMAE: {WMAE: .5f}")
            elif args.model_type == 'convrt':
                df_eval['pred_avg_convrt_'+tar] = np.mean(df_eval['sem_convrt_'+tar+'_l'])
                target_col, pred_col, weight_col = 'sem_convrt_'+tar+'_l', 'pred_avg_convrt_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WMAE: {WMAE: .5f}")  
            elif args.model_type == 'ordersize':
                df_eval['pred_avg_ordersize_'+tar] = np.mean(df_eval['sem_ordersize_'+tar+'_l'])
                target_col, pred_col, weight_col = 'sem_ordersize_'+tar+'_l', 'pred_avg_ordersize_'+tar, 'sem_orders_'+tar+'_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WMAE: {WMAE: .5f}")  
            elif args.model_type == 'rpc':
                df_eval['pred_avg_rpc_new_'+tar] = np.mean(df_eval['sem_rpc_'+tar+'_l'])
                target_col, pred_col, weight_col = 'sem_rpc_'+tar+'_l', 'pred_avg_rpc_new_'+tar, 'sem_clicks_l'
                use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WRMSE: {WRMSE: .5f}")
                logging.info(f"{pred_col.split('_')[2]} oof based on avg WMAE: {WMAE: .5f}")  
        
        logging.info('\n\nevaluating results based on lightgbm for '+tar)
        df_eval['pred_rpc_'+tar] = df_eval['pred_convrt_'+tar] * df_eval['pred_ordersize_'+tar]
        if args.model_type == 'all':
            target_cols = ['sem_convrt_'+tar+'_l', 'sem_ordersize_'+tar+'_l', 'sem_rpc_'+tar+'_l', 'sem_rpc_'+tar+'_l']
            pred_cols = ['pred_convrt_'+tar, 'pred_ordersize_'+tar, 'pred_rpc_'+tar, 'pred_rpc_new_'+tar]
            weight_cols = ['sem_clicks_l', 'sem_orders_'+tar+'_l', 'sem_clicks_l', 'sem_clicks_l']
            use_cols = use_cols + target_cols + pred_cols + weight_cols
            for target_col, pred_col, weight_col in zip(target_cols, pred_cols, weight_cols):
                filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
                WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
                logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WRMSE: {WRMSE: .5f}")
                logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WMAE: {WMAE: .5f}")
        elif args.model_type == 'convrt':
            target_col, pred_col, weight_col = 'sem_convrt_'+tar+'_l', 'pred_convrt_'+tar, 'sem_clicks_l'
            use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
            filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
            WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WRMSE: {WRMSE: .5f}")
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WMAE: {WMAE: .5f}")     
        elif args.model_type == 'ordersize':
            target_col, pred_col, weight_col = 'sem_ordersize_'+tar+'_l', 'pred_ordersize_'+tar, 'sem_orders_'+tar+'_l'
            use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
            filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
            WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WRMSE: {WRMSE: .5f}")
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WMAE: {WMAE: .5f}")   
        elif args.model_type == 'rpc':
            target_col, pred_col, weight_col = 'sem_rpc_'+tar+'_l', 'pred_rpc_new_'+tar, 'sem_clicks_l'
            use_cols = use_cols + [target_col] + [pred_col] + [weight_col]
            filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
            WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WRMSE: {WRMSE: .5f}")
            logging.info(f"{'_'.join(pred_col.split('_')[1:-1])} oof based on lightgbm WMAE: {WMAE: .5f}")   
        # store and upload to gcp
        local_file = f'{args.train_path_local}oof_evaluation_'+tar+'.csv'
        use_cols = list(set(use_cols))
        use_cols.sort()
        df_eval[list(set(use_cols))].to_csv(local_file, index=None)
        gcp_file = f'{args.train_path_gcs}oof_evaluation_'+tar+'.csv'
        upload_local_to_cloud_storage(local_file, gcp_file)

def run_score(args):
    # make sure model type is correct
    if args.model_type not in ['convrt', 'ordersize', 'all', 'rpc']:
        raise RuntimeError(f"Expected model type to be ['convrt', 'ordersize', 'all', 'rpc'], but get '{args.model_type}' instead.")

    # load data from gcp bucket
    config = load_json(args.data_config)
    adid_name = config["hash_id"]["adid"]
    is_mobile_name = config["hash_id"]["is_mobile"]
    source_id_name = config["hash_id"]["source_id"]
    
    #df = read_data_parallel(args.score_path, config["schema"]["testSchema"], cores=16)
    filename = args.path_gcs+'sample_data_full_feat/df_feat_all_coop_item_'+args.ref_date+'.csv'
    df = read_data_csv(filename)
    # prepare test features
    features = config["features"]
    # clean features (all should be numerics)
    features_set = ['uber_curr_item_price_w1', 'uber_avg_overall_rating_w1'] #hard coded for hot fix
    df = convert_numeric(df, features_set)

    # test data filter
    # only score google & bing ads with clicks >= 1 in last 2 week (confidence = 1, 2)
    # test_mask = ((df.sem_clicks_w2 >= 1/14))
    df['confidence'] = assign_confidence(df) 
    df_to_predict = df
    targets = ['seller', 'brand'] if args.target=='all' else [args.target]
    for tar in targets:
        for fold in range(5):
            if args.model_type in ['convrt', 'all']:
                # convrt model
                logging.info('-'*20)
                logging.info(f'Score convrt model fold-{fold}-'+tar+'...')
                model_features = features['convrt'][tar]
                model_name = f'convrt_f{fold}_'+tar
                y_convrt_pred = score(df_to_predict, model_features, args, model_name)

                 # kfold ensemble
                if fold == 0:
                    y_convrt = y_convrt_pred
                else:
                    y_convrt += y_convrt_pred

            if args.model_type in ['ordersize', 'all']:
                # ordersize model
                logging.info('-'*20)
                logging.info(f'Score ordersize model fold-{fold}-'+tar+'...')
                model_features = features['ordersize'][tar]
                model_name = f'ordersize_f{fold}_'+tar
                y_ordersize_pred = score(df_to_predict, model_features, args, model_name)
                # kfold ensemble
                if fold == 0:
                    y_ordersize = y_ordersize_pred
                else:
                    y_ordersize += y_ordersize_pred

            if args.mode == 'score':
                break

        # prepare final results and storage
        df_pred = df_to_predict[config["output_features"][tar]] 
        
        if args.mode == 'score-5fold':
            if args.model_type in ['convrt', 'all']:
                y_convrt /= 5
                df_pred['pred_convrt_'+tar] = y_convrt
            if args.model_type in ['ordersize', 'all']:
                y_ordersize /= 5
                df_pred['pred_ordersize_'+tar] = y_ordersize
            if args.model_type == 'all':
                df_pred['pred_rpc_'+tar] = df_pred['pred_convrt_'+tar] * df_pred['pred_ordersize_'+tar]

        # storage
        local_file = f'{args.score_path_local}scoring_{tar}_{args.model_type}_{args.ref_date}.csv'
        gcp_file = f'{args.score_path_gcs}scoring_{tar}_{args.model_type}_{args.ref_date}.csv'
        df_pred.to_csv(local_file, index=None)
        upload_local_to_cloud_storage(local_file, gcp_file)

        
def main():
    # Input Arguments: argument parsing
    parser = argparse.ArgumentParser(description="SEM rpc prediction model building pipeline.")
    parser.add_argument("--mode", type=str, help="Pipeline mode: train, train-5fold, score, score-5fold")
    parser.add_argument("--model_type", type=str, help="Model type: convrt, ordersize, rpc or all")
    parser.add_argument("--baseline_model", type=bool, default=False, help="baseline model: True or False to train avg results")
    parser.add_argument("--target", type=str, default="all", help="target: brand, seller, all")
    parser.add_argument("--path_gcs", type=str, default="gs://msc_fair_airflow/rpc_model/", help="GCS paths")
    parser.add_argument("--train_path_gcs", type=str, default="gs://msc_fair_airflow/rpc_model/train_path/", help="GCS paths to training data")
    parser.add_argument("--score_path_gcs", type=str, default="gs://msc_fair_airflow/rpc_model/score_path/", help="GCS paths to scoring data")
    parser.add_argument("--train_path_local", type=str, default="./train_output_path/", help="local location to write train related output data")
    parser.add_argument("--score_path_local", type=str, default="./score_output_path/", help="local location to write score related output data")
    parser.add_argument("--output_path_local", type=str, default="./output/", help="Local location to write output data")
    parser.add_argument("--log_filename", type=str, default="rpc_log", help="log file name")
    parser.add_argument("--data_config", type=str, default="gs://msc_fair_airflow/rpc_model/config_rpc.json", help="location of data config json file")
    parser.add_argument("--ref_date", type=str, default="", help="reference date")
    args, _ = parser.parse_known_args()
    
    args.train_path_gcs = args.train_path_gcs+args.ref_date+'/'
    args.score_path_gcs = args.score_path_gcs+args.ref_date+'/'
    
    # check local path
    local_root = Path(args.output_path_local)
    if not local_root.exists():
        local_root.mkdir(exist_ok=True, parents=True)
    train_root = Path(args.train_path_local)
    if not train_root.exists():
        train_root.mkdir(exist_ok=True, parents=True)
    score_root = Path(args.score_path_local)
    if not score_root.exists():
        score_root.mkdir(exist_ok=True, parents=True)

    # handle log filename
    args.log_filename = datetime.now().strftime(f'{args.log_filename}_%Y_%m_%d_%H_%M.log') 
    logging.basicConfig(filename=f'{args.output_path_local}{args.log_filename}', 
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    channel = logging.StreamHandler(sys.stdout)
    sys.stdout = open(f'{args.output_path_local}/{args.log_filename}', 'a', 1)
    channel.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(channel)

    # run pipeline
    logging.info(args)
    mode = args.mode
    logging.info(mode)

    if mode in ['train', 'train-5fold']:
        run_train(args)
    elif mode in ['score', 'score-5fold']:
        run_score(args)
    elif mode == 'train-textTransformer':
        build_text_features(args)
    elif mode == 'score-textTransformer':
        build_text_features(args)
    else:
        logging.info(f'Pipeline mode {mode} is not correct.')
        sys.exit(0)

    # upload log file
    if args.mode.startswith('train'):
        gcp_file = f'{args.train_path_gcs}{args.log_filename}'
    elif args.mode.startswith('score'):
        gcp_file = f'{args.score_path_gcs}{args.log_filename}'
    upload_local_to_cloud_storage(f'{args.output_path_local}{args.log_filename}', gcp_file)

if __name__ == '__main__':
    main()
