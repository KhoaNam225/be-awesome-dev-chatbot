import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings


def init_chat_model():
    boto_session = boto3.Session(
        profile_name="bedrock-developer", region_name="us-east-1"
    )
    sts_client = boto_session.client("sts")

    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::629872170007:role/bedrock-developer",
        RoleSessionName="be-awesome-dev-bedrock-developer",
    )

    credentials = assumed_role["Credentials"]

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name="us-east-1",
    )

    llm = ChatBedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        region_name="us-east-1",
        client=bedrock_client,
    )

    return llm


def init_embedding_model():
    boto_session = boto3.Session(
        profile_name="bedrock-developer", region_name="us-east-1"
    )
    sts_client = boto_session.client("sts")

    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::629872170007:role/bedrock-developer",
        RoleSessionName="be-awesome-dev-bedrock-developer",
    )

    credentials = assumed_role["Credentials"]

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name="us-east-1",
    )

    embedding_model = BedrockEmbeddings(
        client=bedrock_client,
        region_name="us-east-1",
        model_id="amazon.titan-embed-text-v2:0",
    )

    return embedding_model
