import random
from google.cloud import storage
import os 
import string
import requests

# def request_hackaton(api, filename):
    
#     files = {"filename":filename}
#     r = requests.post(url=api, json=files)
#     response = r.json() if r.status_code == 200 else { "status_code":  r.status_code}
#     return response


# if __name__ == "__main__":
#     API = "http://localhost:6004/hackaton"
#     filename = "data.csv"

#     print(request_hackaton(API, filename))


def request_hackaton(api, content):
    
    data = {"content":content}
    r = requests.post(url=api, json=data)
    response = r.json() if r.status_code == 200 else { "status_code":  r.status_code}
    return response


if __name__ == "__main__":
    API = "http://localhost:6004/get_claims"
    content = [{ 'content': 'the war in ukraine is a hoax' }]

    print(request_hackaton(API, content))