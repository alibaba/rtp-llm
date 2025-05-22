import os
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from io import BytesIO

def get_bytes_io_from_oss_path(url: str):
    if not url.startswith("oss://"):
        raise Exception(f"oss path format expect start with oss://, actual: {url}")
    url = url.replace("oss://", "")    
    bucket_name, path = url.split('/', 1)
    endpoint = os.environ['OSS_ENDPOINT']
    # The AccessKey pair of an Alibaba Cloud account has permissions on all API operations. Using these credentials to perform operations in OSS is a high-risk operation. We recommend that you use a RAM user to call API operations or perform routine O&M. To create a RAM user, log on to the RAM console. 
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    # In this example, the endpoint of the China (Hangzhou) region is used. Specify your actual endpoint. 
    # Specify the name of the bucket. Example: examplebucket. 
    bucket = oss2.Bucket(auth, endpoint, bucket_name, connect_timeout=100)
    object = bucket.get_object(path)
    io = BytesIO(object.read())
    return io


