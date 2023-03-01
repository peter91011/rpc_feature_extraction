import pandas as pd
import numpy as np
import pandas as pd
import re, json
from google.cloud import storage

def reduce_mem_usage(props):
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(0,inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

    return props, NAlist

def convert_numeric(df, features):
    for feat in features:
        if feat in df.columns:
            df[feat] = df[feat].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    return df

## GCP related
def extract_gcs_path(path):
    # we will always assume the object path should be in
    # gs://bucket/folder/.../file
    # format
    m = re.match(r"^gs://(.*?)/(.+)", path)

    if m:
        return m.group(1), m.group(2)

    raise Exception("failed to path gcs path %s to extract bucket and path" % path)

def extract_gcs_files(gcp_file_path):
    # we will always assume the object path should be in
    # gs://bucket/folder/.../file
    # format
    # this function returns a list of all files in the given path
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=object_path)
    files = []
    for blob in blobs:
        if blob.name.endswith('/') or blob.name.endswith('_SUCCESS'):
            pass
        else:
            files.append(f'gs://{bucket_name}/{blob.name}')
    return files

def create_gcp_directory(gcp_file_path):
    """
    create gcp fie path, we assume the path should be in
    gs://bucket/some/folder/path/
    format
    """
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = storage.blob.Blob(object_path, bucket)
    if blob.exists():
        return
    else:
        blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

def upload_local_to_cloud_storage(local_file_path, gcp_file_path):
    """Upload file to GCP bucket"""
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = storage.blob.Blob(object_path, bucket)
    blob.upload_from_filename(local_file_path)


def download_from_cloud_storage(local_file_path, gcp_file_path):
    """Download file from GCP bucket"""
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = storage.blob.Blob(object_path, bucket)
    blob.download_to_filename(local_file_path)    
    
def upload_csv_to_cloud_storage(df, gcp_file_path):
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(gcp_file_path.replace('gs://','').replace(bucket_name+'/','')).upload_from_string(df.to_csv(index=False), 'text/csv')

def delete_from_cloud_storage(gcp_file_path):
    storage_client = storage.Client()
    bucket_name, object_path = extract_gcs_path(gcp_file_path)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=object_path)
    for blob in blobs:
        blob.delete()

def load_json(path):
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    except:
        if path.startswith("gs://"):
            bucket_name, object_path = extract_gcs_path(path)
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = storage.blob.Blob(object_path, bucket)
            data = json.loads(blob.download_as_string())
            return data
        else:
            raise RuntimeError(f'{path} is neither local path nor gcp path')