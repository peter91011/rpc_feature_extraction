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

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from utils.utils import (reduce_mem_usage, 
                         extract_gcs_path, 
                         extract_gcs_files, 
                         create_gcp_directory, 
                         upload_local_to_cloud_storage, 
                         download_from_cloud_storage, 
                         load_json,
                         convert_numeric
                         )

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

def wrmse(y, y_pred, weights):
    error = np.power(y - y_pred, 2)*weights
    return np.sqrt(np.sum(error)/np.sum(weights))

def wmae(y, y_pred, weights):
    error = np.abs(y - y_pred)*weights
    return np.sum(error)/np.sum(weights)

# build text features
def text_transformer(data, prefix, output_dim=32, vec=None, svd=None):
    # tf-idf conversion
    if not vec:
        vec = TfidfVectorizer(ngram_range=(1,3),
                              stop_words='english',
                              strip_accents='unicode',
                              max_features=100000)
        vec.fit(data.tolist())
    # transform tf-idf
    X = vec.transform(data)

    # dimension reduction
    if not svd:
        svd = TruncatedSVD(n_components=output_dim, random_state=1337)
        svd.fit(X)
    logging.info(f'SVD variance ratio sum: {svd.explained_variance_ratio_.sum():.5f}')
    logging.info(f"SVD variance ratio: {''.join([str(x) for x in svd.explained_variance_ratio_])}")
    # transform svd
    X_svd = svd.transform(X)
    # store results
    df_text = pd.DataFrame(X_svd)
    df_text = df_text.add_prefix(prefix)
    return df_text, vec, svd

def build_text_features(args):
    # load data from gcp bucket, get adid list
    config = load_json(args.data_config)
    adid_name = config["hash_id"]["adid"]

    if args.mode.startswith('train'):
        df = read_data_parallel(args.train_path, config["schema"]["trainSchema"], cores=16)
        adids = df[df[adid_name].notnull()][adid_name].astype(str).tolist()
    elif args.mode.startswith('score'):
        df = read_data_parallel(args.score_path, config["schema"]["testSchema"], cores=16)
        adids = df[df[adid_name].notnull()][adid_name].astype(str).tolist()
    adids = list(set(adids))
    df_ref = pd.DataFrame({adid_name: adids})
    logging.info(f'Number of ads to be processed: {df_ref.shape[0]}')

    # spark related
    spark = SparkSession.builder.config("spark.kryoserializer.buffer.max",
                                     "2000m").enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    sqlContext = HiveContext(sc)
    sqlContext.setConf("hive.exec.dynamic.partition", "true")
    sqlContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    mySchema = StructType([StructField(adid_name, StringType(), True)])
    df_ref_spark = spark.createDataFrame(df_ref, schema=mySchema)
    df_ref_spark.createOrReplaceTempView('tmp_table')

    # query text related features
    partitions = sqlContext.sql("SHOW partitions dmas.product_attrs").toPandas()
    latest_ts = partitions.partition.iloc[-1].split('=')[1]
    logging.info(f'Load text featres from dmas.product_attrs partition: {latest_ts}')
    sql_cmd = f"""
        SELECT t1.{adid_name},
               t3.item_name,
               t3.item_short_desc,
               t3.item_long_desc
          FROM (
       (SELECT * FROM tmp_table) t1
    INNER JOIN sem.wmt_dotcom_items t2
            ON (t1.{adid_name}=t2.id)
    INNER JOIN
       (SELECT *
          FROM dmas.product_attrs
         WHERE ts='{latest_ts}') t3
            ON (t2.catalog_item_id=t3.item_id))
    """
    df_text = sqlContext.sql(sql_cmd).toPandas()
    logging.info(f'Text df shape: {df_text.shape}')
    # remove duplicate
    df_text = df_text.drop_duplicates(subset=adid_name, keep='first')
    logging.info(f'Text df shape after dropping duplicates: {df_text.shape}')

    # transfer text features
    df_text[adid_name] = df_text[adid_name].astype(int)
    df_text.fillna('null', inplace=True)
    # title
    if args.mode.startswith('train'):
        vec_title, svd_title, vec_desc, svd_desc = None, None, None, None
    elif args.mode.startswith('score'):
        # download to local from gcp
        download_from_cloud_storage(f"{args.local_path}/vec_title.pkl", f"{args.train_output_path}vec_title.pkl")
        download_from_cloud_storage(f"{args.local_path}/svd_title.pkl", f"{args.train_output_path}svd_title.pkl")
        download_from_cloud_storage(f"{args.local_path}/vec_desc.pkl", f"{args.train_output_path}vec_desc.pkl")
        download_from_cloud_storage(f"{args.local_path}/svd_desc.pkl", f"{args.train_output_path}svd_desc.pkl")
        # load from pretrained model
        with open(f"{args.local_path}/vec_title.pkl", "rb") as f:
            vec_title = pickle.load(f)
        with open(f"{args.local_path}/svd_title.pkl", "rb") as f:
            svd_title = pickle.load(f)
        with open(f"{args.local_path}/vec_desc.pkl", "rb") as f:
            vec_desc = pickle.load(f)
        with open(f"{args.local_path}/svd_desc.pkl", "rb") as f:
            svd_desc = pickle.load(f)

    start_time = time.time()
    logging.info(f'Building item_name feautres...')
    X_title, vec_title, svd_title = text_transformer(df_text['item_name'], prefix='item_name_', output_dim=32, vec=vec_title, svd=svd_title)
    time_elapse = (time.time()-start_time)/60
    logging.info(f'Finished buliding item_name features, took {time_elapse:.2f} mins.')
    # short_desc
    start_time = time.time()
    logging.info(f'Building short_desc feautres...')
    X_short_desc, vec_desc, svd_desc = text_transformer(df_text['item_short_desc'], prefix='item_short_desc_', output_dim=64, vec=vec_desc, svd=svd_desc)
    time_elapse = (time.time()-start_time)/60
    logging.info(f'Finished buliding item_short_desc features, took {time_elapse:.2f} mins.')
    # sanity check
    assert (df_text.shape[0] == X_title.shape[0])
    assert (df_text.shape[0] == X_short_desc.shape[0])
    df_text = pd.concat([df_text, X_title, X_short_desc], axis=1, sort=False)
    logging.info(f'Finished generate text features, data shape: {df_text.shape}.')
    # save transformer and svd
    if args.mode.startswith('train'):
        # dump to local
        pickle.dump(vec_title, open(f"{args.local_path}/vec_title.pkl", "wb"))
        pickle.dump(svd_title, open(f"{args.local_path}/svd_title.pkl", "wb"))
        pickle.dump(vec_desc, open(f"{args.local_path}/vec_desc.pkl", "wb"))
        pickle.dump(svd_desc, open(f"{args.local_path}/svd_desc.pkl", "wb"))
        # upload to gcp
        upload_local_to_cloud_storage(f"{args.local_path}/vec_title.pkl", f"{args.train_output_path}vec_title.pkl")
        upload_local_to_cloud_storage(f"{args.local_path}/svd_title.pkl", f"{args.train_output_path}svd_title.pkl")
        upload_local_to_cloud_storage(f"{args.local_path}/vec_desc.pkl", f"{args.train_output_path}vec_desc.pkl")
        upload_local_to_cloud_storage(f"{args.local_path}/svd_desc.pkl", f"{args.train_output_path}svd_desc.pkl")

    # storage
    if args.mode.startswith('train'):
        output_path = args.train_output_path
    elif args.mode.startswith('score'):
        output_path = args.score_output_path

    local_file = f'{args.local_path}/{args.text_feature_file}'
    gcp_file = f'{output_path}{args.text_feature_file}'
    # store to local
    start_time = time.time()
    df_text.dropna(subset=[adid_name], inplace=True) #remove NAN items
    df_text.to_csv(local_file, index=None, encoding="utf-8")
    time_elapse = (time.time()-start_time)/60
    logging.info(f'Store features to local, took {time_elapse:.2f} mins.')
    # upload to gcp
    start_time = time.time()
    upload_local_to_cloud_storage(local_file, gcp_file)
    time_elapse = (time.time()-start_time)/60
    logging.info(f'Upload features to gcp, took {time_elapse:.2f} mins.')
    return df_text

def train(df_trn, df_val, feature_cols, target_col, lgb_params, args, model_name, weight_col=None):
    # build dataset
    if weight_col:
        dtrain = lgb.Dataset(df_trn[feature_cols],
                             df_trn[target_col].values,
                             weight=df_trn[weight_col].values,
                             free_raw_data=False,
                             silent=False,
                             )
        dvalid = lgb.Dataset(df_val[feature_cols],
                             df_val[target_col].values,
                             weight=df_val[weight_col].values,
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

    # save model to local path
    local_file = f'{args.local_path}/{model_name}.txt'
    model.save_model(local_file, num_iteration=model.best_iteration)

    # upload model to gcp
    gcp_file = f'{args.train_output_path}{model_name}.txt'
    upload_local_to_cloud_storage(local_file, gcp_file)

def eval(df, feature_cols, target_col, args, model_name, weight_col=None):
    try:
        model = lgb.Booster(model_file=(f'{args.local_path}/{model_name}.txt'))
    except:
        raise RuntimeError(f'{model_name} does not exist...')

    y_pred = model.predict(df[feature_cols])
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
    local_file = f'{args.local_path}/{model_name}.txt'
    gcp_file = f'{args.train_output_path}{model_name}.txt'
    download_from_cloud_storage(local_file, gcp_file)

    # load model and score
    try:
        model = lgb.Booster(model_file=local_file)
    except:
        raise RuntimeError(f'{local_file} does not exist...')
    return model.predict(df[feature_cols])

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
    if args.model_type not in ['convrt', 'ordersize', 'all']:
        raise RuntimeError(f"Expected model type to be ['convrt', 'ordersize' or 'both'], but get '{args.model_type}' instead.")

    # lgb paramters
    lgb_params = {
        'n_estimators': 5000,
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
    df = read_data_parallel(args.train_path, config["schema"]["trainSchema"], cores=16)

    # prepare train features
    features = config["features"]
    # clean features (all should be numerics)
    features_set = ['uber_curr_item_price_w1', 'uber_avg_overall_rating_w1'] #hard coded for hot fix
    # for f in features.values():
    #     features_set += f
    # features_set = list(set(features_set))
    df = convert_numeric(df, features_set)
    
    # text features
    if args.flag_text_features:
        # load text features
        local_text_file = f'{args.local_path}/{args.text_feature_file}'
        df_text = pd.read_csv(local_text_file)
        df_text.dropna(subset=[adid_name], inplace=True) #remove nan items

        # merge text features
        # check merge col type
        if df[adid_name].dtype != object:
            df[adid_name] = df[adid_name].astype(int).astype(str)
        if df_text[adid_name].dtype != object:
            df_text[adid_name] = df_text[adid_name].astype(int).astype(str)     
        df = pd.merge(df, df_text, how='left', on=adid_name)
        logging.info(f'Merged text feature, data shape: {df.shape}')

    # do train-val split
    df['fold'] = 0
    kf = KFold(5, shuffle=True, random_state=42)
    for i, (_, val_idx) in enumerate(kf.split(df)):
        df['fold'].iloc[val_idx] = i

    # filters
    convrt_trn_filter = (df[source_id_name]==2) & (df.sem_convrt_l>=0) & (df.sem_convrt_l<=0.3)
    convrt_val_filter = (df[source_id_name]==2) & (df.sem_convrt_l>=0) & (df.sem_convrt_l<=0.3)
    ordersize_trn_filter = (df[source_id_name]==2) & (df.sem_orders_l>=3) & (df.sem_ordersize_l>=0) & (df.sem_ordersize_l<=500)
    ordersize_val_filter = (df[source_id_name]==2) & (df.sem_ordersize_l>=0) & (df.sem_ordersize_l<=500)

    # do train & eval
    # prepare dataframe to store OOF results
    df_eval = df.copy()
    if args.model_type == 'all':
        df_eval['pred_convrt'] = np.nan
        df_eval['pred_ordersize'] = np.nan
        df_eval['pred_rpc'] = np.nan
    elif args.model_type == 'convrt':
        df_eval['pred_convrt'] = np.nan
    elif args.model_type == 'ordersize':
        df_eval['pred_ordersize'] = np.nan

    for fold in range(5):
        if args.model_type in ['convrt', 'all']:
            # convrt model
            logging.info('-'*20)
            logging.info(f'Train convrt model fold-{fold}...')
            model_features = features['convrt']
            target_col = "sem_convrt_l"
            model_name = f'convrt_f{fold}'
            trn_mask = convrt_trn_filter & (df.fold != fold)
            df_trn = df[trn_mask]
            val_mask = convrt_val_filter & (df.fold == fold)
            df_val = df[val_mask]
            train(df_trn, df_val, model_features, target_col, lgb_params, args, model_name, weight_col='sem_clicks_l')

            # Evaluate
            logging.info(f'Eval fold-{fold} results...')
            y_pred = eval(df_val, model_features, target_col, args, model_name, weight_col='sem_clicks_l')
            df_eval['pred_convrt'].loc[val_mask] = y_pred

        if args.model_type in ['ordersize', 'all']:
            # order_size model
            logging.info('-'*20)
            logging.info(f'Train ordersize model fold-{fold}...')
            model_features = features['ordersize']
            target_col = "sem_ordersize_l"
            model_name = f'ordersize_f{fold}'
            trn_mask = ordersize_trn_filter & (df.fold != fold)
            df_trn = df[trn_mask]
            val_mask = ordersize_val_filter & (df.fold == fold)
            df_val = df[val_mask]
            train(df_trn, df_val, model_features, target_col, lgb_params, args, model_name, weight_col='sem_orders_l')

            # Evaluate
            logging.info(f'Evaluate fold-{fold} results...')
            y_pred = eval(df_val, model_features, target_col, args, model_name, weight_col='sem_orders_l')
            df_eval['pred_ordersize'].loc[val_mask] = y_pred

        if args.mode == "train":
            break
    
    # report and store oof results
    logging.info('Evaluate out of fold results')
    df_eval['pred_rpc'] = df_eval['pred_convrt'] * df_eval['pred_ordersize']
    if args.model_type == 'all':
        target_cols = ['sem_convrt_l', 'sem_ordersize_l', 'sem_rpc_l']
        pred_cols = ['pred_convrt', 'pred_ordersize', 'pred_rpc']
        weight_cols = ['sem_clicks_l', 'sem_orders_l', 'sem_clicks_l']
        use_cols = target_cols + pred_cols + weight_cols
        for target_col, pred_col, weight_col in zip(target_cols, pred_cols, weight_cols):
            filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
            WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
            logging.info(f"{target_col.split('_')[1]} oof WRMSE: {WRMSE: .5f}")
            logging.info(f"{target_col.split('_')[1]} oof WMAE: {WMAE: .5f}")
    elif args.model_type == 'convrt':
        target_col, pred_col, weight_col = 'sem_convrt_l', 'pred_convrt', 'sem_clicks_l'
        use_cols = [target_col, pred_col, weight_col]
        filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
        WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
        WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
        logging.info(f"{target_col.split('_')[1]} oof WRMSE: {WRMSE: .5f}")
        logging.info(f"{target_col.split('_')[1]} oof WMAE: {WMAE: .5f}")     
    elif args.model_type == 'ordersize':
        target_col, pred_col, weight_col = 'sem_ordersize_l', 'pred_ordersize', 'sem_orders_l'
        use_cols = [target_col, pred_col, weight_col]
        filter_NoNA = df_eval[target_col].notnull() & df_eval[pred_col].notnull() & df_eval[weight_col].notnull()
        WRMSE = wrmse(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
        WMAE = wmae(df_eval[target_col].loc[filter_NoNA].values, df_eval[pred_col].loc[filter_NoNA].values, df_eval[weight_col].loc[filter_NoNA].values)
        logging.info(f"{target_col.split('_')[1]} oof WRMSE: {WRMSE: .5f}")
        logging.info(f"{target_col.split('_')[1]} oof WMAE: {WMAE: .5f}")   
    # store and upload to gcp
    local_file = f'{args.local_path}/oof_evaluation.csv'
    gcp_file = f'{args.train_output_path}oof_evaluation.csv'
    use_cols = list(set(use_cols))
    df_eval[use_cols].to_csv(local_file, index=None)
    upload_local_to_cloud_storage(local_file, gcp_file)

    # mark train success
    if args.flag_check_train_success:
        if args.flag_train_success:
            if args.model_type in ['convrt', 'all']:
                # mark convrt success
                local_file = args.local_path + '/' + args.convrt_success_path.split('/')[-1]
                with open(local_file, 'w') as fp:
                    pass
                upload_local_to_cloud_storage(local_file, args.convrt_success_path)
            if args.model_type in ['ordersize', 'all']:
                # mark ordersize success
                local_file = args.local_path + '/' + args.ordersize_success_path.split('/')[-1]
                with open(local_file, 'w') as fp:
                    pass                
                upload_local_to_cloud_storage(local_file, args.ordersize_success_path)

def run_score(args):
    # make sure model type is correct
    if args.model_type not in ['convrt', 'ordersize', 'all']:
        raise RuntimeError(f"Expected model type to be ['convrt', 'ordersize' or 'both'], but get '{args.model_type}' instead.")

    # load data from gcp bucket
    config = load_json(args.data_config)
    adid_name = config["hash_id"]["adid"]
    is_mobile_name = config["hash_id"]["is_mobile"]
    source_id_name = config["hash_id"]["source_id"]
    df = read_data_parallel(args.score_path, config["schema"]["testSchema"], cores=16)
    
    # prepare test features
    features = config["features"]
    # clean features (all should be numerics)
    features_set = ['uber_curr_item_price_w1', 'uber_avg_overall_rating_w1'] #hard coded for hot fix
    # for f in features.values():
    #     features_set += f
    # features_set = list(set(features_set))
    df = convert_numeric(df, features_set)

    # text features
    if args.flag_text_features:
        start_time=time.time()
        local_text_file = f'{args.local_path}/{args.text_feature_file}'
        df_text = pd.read_csv(local_text_file)
        df_text.dropna(subset=[adid_name], inplace=True) #remove nan items

        # merge text features
        # check merge col type
        if df[adid_name].dtype != object:
            df[adid_name] = df[adid_name].astype(int).astype(str)
        if df_text[adid_name].dtype != object:
            df_text[adid_name] = df_text[adid_name].astype(int).astype(str)     
        df = pd.merge(df, df_text, how='left', on=adid_name)
        coverage = df['item_name'].isnull()
        logging.info(f'Missing: {coverage.sum()}; pct: {coverage.mean():.2f}.')

    # test data filter
    # only score google & bing ads with clicks >= 1 in last 2 week (confidence = 1, 2)
    # test_mask = ((df.sem_clicks_w2 >= 1/14))
    df['confidence'] = assign_confidence(df) 
    df_to_predict = df
    
    for fold in range(5):
        if args.model_type in ['convrt', 'all']:
            # convrt model
            logging.info('-'*20)
            logging.info(f'Score convrt model fold-{fold}...')
            model_features = features['convrt']
            model_name = f'convrt_f{fold}'
            y_convrt_pred = score(df_to_predict, model_features, args, model_name)

             # kfold ensemble
            if fold == 0:
                y_convrt = y_convrt_pred
            else:
                y_convrt += y_convrt_pred

        if args.model_type in ['ordersize', 'all']:
            # ordersize model
            logging.info('-'*20)
            logging.info(f'Score ordersize model fold-{fold}...')
            model_features = features['ordersize']
            model_name = f'ordersize_f{fold}'
            y_ordersize_pred = score(df_to_predict, model_features, args, model_name)
            # kfold ensemble
            if fold == 0:
                y_ordersize = y_ordersize_pred
            else:
                y_ordersize += y_ordersize_pred

        if args.mode == 'score':
            break

    # prepare final results and storage
    df_pred = df_to_predict[config["output_features"]] 
    
    if args.mode == 'score-5fold':
        if args.model_type in ['convrt', 'all']:
            y_convrt /= 5
            df_pred['pred_convrt'] = y_convrt
        if args.model_type in ['ordersize', 'all']:
            y_ordersize /= 5
            df_pred['pred_ordersize'] = y_ordersize
        if args.model_type == 'all':
            df_pred['pred_rpc'] = df_pred['pred_convrt'] * df_pred['pred_ordersize']

    # storage
    local_file = f'{args.local_path}/scoring_{args.model_type}.csv'
    gcp_file = f'{args.score_output_path}scoring_{args.model_type}.csv'
    df_pred.to_csv(local_file, index=None)
    upload_local_to_cloud_storage(local_file, gcp_file)

def main():
    # Input Arguments: argument parsing
    parser = argparse.ArgumentParser(description="SEM rpc prediction model building pipeline.")
    parser.add_argument("--mode", type=str, help="Pipeline mode: train, train-5fold, score, score-5fold")
    parser.add_argument("--model_type", type=str, help="Model type: convrt, ordersize or all")
    parser.add_argument("--train_path", type=str, default="", help="GCS paths to training data")
    parser.add_argument("--score_path", type=str, default="", help="GCS paths to scoring data")
    parser.add_argument("--train_output_path", type=str, default="", help="GCS location to write train related output data")
    parser.add_argument("--score_output_path", type=str, default="", help="GCS location to write score related output data")
    parser.add_argument("--local_path", type=str, default="./output", help="Local location to write output data")
    parser.add_argument("--log_filename", type=str, default="logfile", help="log file name")
    parser.add_argument("--data_config", type=str, default="./config.json", help="location of data config json file")
    parser.add_argument("--flag_text_features", type=int, default=0, help="Add text features or not.")
    parser.add_argument("--flag_train_text_transformer", type=int, default=0, help="Flag train text transfomer or not.")
    parser.add_argument("--text_feature_file", type=str, default="text_feature.csv", help="text features file name")
    parser.add_argument("--flag_check_train_success", type=bool, default=False, help="Flag for check train success or not.")
    parser.add_argument("--flag_train_success", type=bool, default=True, help="Flag for train success or not")
    parser.add_argument("--ordersize_success_path", type=str, default="", help="Ordersize success mark path")
    parser.add_argument("--convrt_success_path", type=str, default="", help="Convrt success mark path")
    args, _ = parser.parse_known_args()

    # check local path
    local_root = Path(args.local_path)
    if not local_root.exists():
        local_root.mkdir(exist_ok=True, parents=True)

    # handle log filename
    args.log_filename = datetime.now().strftime(f'{args.log_filename}_%H_%M_%d_%m_%Y.log') 
    logging.basicConfig(filename=f'{args.local_path}/{args.log_filename}', 
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    channel = logging.StreamHandler(sys.stdout)
    sys.stdout = open(f'{args.local_path}/{args.log_filename}', 'a', 1)
    channel.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(channel)

    # create gcp directory in case it does not exist
    if args.train_output_path:
        create_gcp_directory(args.train_output_path)
    if args.score_output_path:
        create_gcp_directory(args.score_output_path)

    # run pipeline
    logging.info(args)
    mode = args.mode
    logging.info(mode)

    # this is temperoary, after fully migrate to GCP we will remove this part.
    if args.flag_text_features:
        build_text_features(args)

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
        gcp_file = f'{args.train_output_path}{args.log_filename}'
    elif args.mode.startswith('score'):
        gcp_file = f'{args.score_output_path}{args.log_filename}'
    upload_local_to_cloud_storage(f'{args.local_path}/{args.log_filename}', gcp_file)

if __name__ == '__main__':
    main()
