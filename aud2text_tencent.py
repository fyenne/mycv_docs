import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import base64 
import re




class Keys:
    def __init__(self) -> None:
        self.SecretId = 'AKIDufDXcS3PWO1PYS7XidJkyyjGYbPyJSKw'
        self.SecretKey= 'QbZJo44pLceRzIMCnAIAnCMHxOZVCerp'
        self.cred = credential.Credential(self.SecretId, self.SecretKey)

        

# file = "/Users/fyenne/Downloads/download.wav"
# enc = (base64.b64encode(open(file, "rb").read()))
 
# with open("/Users/fyenne/Downloads/outp.txt", 'wb') as a:
#     a.write(enc)
def file_in(path):
    with open("/Users/fyenne/Downloads/outp.txt", 'r') as f:
        enc = f.read() 

    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户secretId，secretKey,此处还需注意密钥对的保密
        # 密钥可前往https://console.cloud.tencent.com/cam/capi网站进行获取
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = asr_client.AsrClient(cred, "", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.CreateRecTaskRequest()
        params = {
            "EngineModelType": "16k_zh_video",
            "ChannelNum": 1,
            "SpeakerDiarization": 1,
            "ResTextFormat": 3,
            "SourceType": 1,
            "Data":enc
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个CreateRecTaskResponse的实例，与请求对象对应
        resp = client.CreateRecTask(req)
        # 输出json格式的字符串回包
        print(resp.to_json_string())
        result_to_check = json.loads(resp.to_json_string())['Data']['TaskId']
    
    # {"Data": {"TaskId": 2249976223}, "RequestId": "f9c7419b-0793-4f46-a0b8-e0cfb99c07d9"}
    except TencentCloudSDKException as err:
        print(err)
    
    return result_to_check



def file_out(result_to_check):
    req = models.DescribeTaskStatusRequest()
    params = {
        'TaskId': result_to_check

    }
    req.from_json_string(json.dumps(params))

    # 返回的resp是一个DescribeTaskStatusResponse的实例，与请求对象对应
    resp = client.DescribeTaskStatus(req)
    # 输出json格式的字符串回包
    print(resp.to_json_string())
    return resp.to_json_string()