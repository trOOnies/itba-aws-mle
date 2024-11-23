"""
MIT No Attribution

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import boto3
import json
import os

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
runtime = boto3.client("runtime.sagemaker")


def build_response(status_code, response_body):
    print(status_code)
    print(response_body)

    response = {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "*",
        },
    }

    if response_body is not None:
        response["body"] = str(response_body["predictions"][0]["score"])

    return response


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    if "requestContext" in event:
        if event["httpMethod"] == "OPTIONS":
            return build_response(200, "")

        elif event["httpMethod"] == "POST":
            turbine_data = event["body"]

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=turbine_data
            )
            print(response)
            result = json.loads(response["Body"].read().decode())
            print(result)
            return build_response(200, result)

        else:
            return build_response(405, None)
