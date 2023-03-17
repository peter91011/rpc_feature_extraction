from pyspark.sql import functions as func
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext, SparkSession
from pyspark.sql.functions import col, struct, to_json, udf, expr
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

import time, json, argparse, logging, sys
from datetime import datetime, timedelta
from tqdm.notebook import tqdm

def upload(args):
    spark = SparkSession.builder.config("spark.kryoserializer.buffer.max",
                                     "2000m").enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    sqlContext = HiveContext(sc)
    sqlContext.setConf("hive.exec.dynamic.partition", "true")
    sqlContext.setConf("hive.exec.dynamic.partition.mode", "nonstrict")

    # load prediction file
    logging.info(f"Load prediction table {args.file_path}")
    df = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(args.file_path)
    df = df.withColumn("rpc", df["pred_convrt"]*df["pred_ordersize"])
    df = df.withColumn('pred_convrt1', df['pred_convrt'])
    df = df.select("sem_adid_w2", "sem_is_mobile_w2", "uber_division_w1", "rpc", "pred_convrt1", "pred_ordersize")
    logging.info(f"Prediction table has {df.count()} rows.")
    
    # load ab test file & generate ads to be process
    if args.abtest_file:
        logging.info(f"Load ab-test table {args.abtest_file}")
        df_ab = spark.read.format("csv").option("header", "true").option("mode", "DROPMALFORMED").load(args.abtest_file)
        df_ab = df_ab.toPandas()
        adid_list = df_ab[df_ab.group=='1'].adid.astype(str).tolist()
        adid_list = set(adid_list)
        logging.info(f"AB-test ads {df_ab.shape[0]} in total, {len(adid_list)} in experiment group.")
    else:
        logging.info(f"No ab-test")
        adid_list = None
        
    # load production table to be modified
    logging.info(f"Update {args.prod_table} partition ds={args.date}")    
    sql_cmd = f"""
        SELECT * FROM {args.prod_table}
                WHERE ds='{args.date}'
                  AND attr_meth='LTA'
                  AND model='warm'
    """    
    df_prod = sqlContext.sql(sql_cmd)
    
    # merge tables and modify
    # helper functions
    args.division = args.division.split(',') if args.division else None
    args.super_dept = args.super_dept.split(',') if args.super_dept else None
        
    def f_rpc(adjusted_rpc, rpc, division, super_dept, adid):
        # ab test mode, only modify ads in the adid_list
        if adid_list and str(adid) not in adid_list:
            return adjusted_rpc
        # only modify those rows in the filter
        if args.division and division not in args.division:
            return adjusted_rpc
        if args.super_dept and super_dept not in args.super_dept:
            return adjusted_rpc
        # write rpc to adjusted_rpc if rpc is not null
        if rpc:
            return rpc
        else:
            return adjusted_rpc
    udf_rpc = udf(f_rpc)

    def f_adj_reason(adj_reason, rpc, pred_convrt, pred_ordersize, division, super_dept, adid):
        # ab test mode, only modify ads in the adid_list
        if adid_list and str(adid) not in adid_list:
            return adj_reason
        # only modify those rows in the filter
        if args.division and division not in args.division:
            return adj_reason
        if args.super_dept and super_dept not in args.super_dept:
            return adj_reason
        # add adj_reason if rpc is not null
        if rpc:
            res = {
                "objective": "Experiment",
                "convrt": pred_convrt,
                "ordersize": pred_ordersize
            }
            return json.dumps(res)
        else:
            return adj_reason
    udf_adj_reason = udf(f_adj_reason)
    
    # join two tables and modify rows
    ## TODO: fix duplicate issue. Temp solution: add hierarchy as key.
    new_df = df_prod.join(df, (df_prod.adid==df.sem_adid_w2) & (df_prod.is_mobile==df.sem_is_mobile_w2) & (df_prod.division==df.uber_division_w1), "left")
    new_df = new_df.withColumn("adjusted_rpc1", udf_rpc("adjusted_rpc", "rpc", "division", "super_dept", "adid"))
    new_df = new_df.withColumn("adj_reason", udf_adj_reason("adj_reason", "rpc", "pred_convrt1", "pred_ordersize", "division", "super_dept", "adid"))
    n_modified = new_df.filter(new_df['adjusted_rpc'] != new_df['adjusted_rpc1']).count()
    new_df = new_df.withColumn("adjusted_rpc", new_df["adjusted_rpc1"])
    logging.info(f"Modified {n_modified} ads.")
    
    # write new_df to production table
    logging.info("Upload modified table. ")
    tmp_table = "tmp_table"
    new_df.createOrReplaceTempView(tmp_table)
    logging.info(f"Product table rows: {df_prod.count()}, new table rows: {new_df.count()}.")
    assert (new_df.count() == df_prod.count())
    sql_cmd = f"""
        INSERT OVERWRITE TABLE {args.prod_table}
        PARTITION(ds='{args.date}', attr_meth='LTA', model='warm', confidence)
        SELECT
            adid,
            keyword_external_id,
            account_external_id,
            campaign_external_id,
            group_external_id,
            catalog_item_id,
            source_id,
            is_mobile,
            division,
            super_dept,
            dept,
            category,
            sub_category,
            clicks_l2wk,
            conv_l2wk,
            rpc_l2wk,
            pred_convrt,
            pred_os,
            pred_rpc,
            adjusted_rpc,
            adj_reason,
            hierarchy_rpc_ratio,
            confidence
        FROM {tmp_table}
        WHERE
            ds='{args.date}'
            AND attr_meth='LTA'
            AND model='warm'
    """
    sqlContext.sql(sql_cmd)
    logging.info("Finished upload updated results.")
    
def main():
    # Input Arguments: argument parsing
    parser = argparse.ArgumentParser(description='Upload rpc prediction to production table.')
    parser.add_argument('--prod_table', type=str, help='Prodution table name')
    parser.add_argument('--date', type=str, help='Prodution table date')
    parser.add_argument('--file_path', type=str, help='File path to upload')
    parser.add_argument('--abtest_file', type=str, default='', help='AB testing grouping info table path')
    parser.add_argument('--division', type=str, default='', help="Divisions to be modified, seperated by ','")
    parser.add_argument('--super_dept', type=str, default='', help="Super departments to be modified, seperated by ','")
    parser.add_argument('--log_filename', type=str, default='logfile', help='log file name')
    args, _ = parser.parse_known_args()

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

    # run pipeline
    logging.info(args)
    upload(args)

if __name__ == '__main__':
    main()
