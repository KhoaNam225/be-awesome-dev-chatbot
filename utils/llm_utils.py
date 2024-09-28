import boto3
from langchain_aws import AmazonKnowledgeBasesRetriever, ChatBedrock, BedrockEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

LLM = None


def init_chat_model():
    global LLM

    if LLM is not None:
        return LLM

    boto_session = boto3.Session(region_name="us-east-1")
    sts_client = boto_session.client("sts")

    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::629872170007:role/bedrock-consumer",
        RoleSessionName="be-awesome-dev-bedrock-consumer",
    )

    credentials = assumed_role["Credentials"]

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name="us-east-1",
    )

    LLM = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        region_name="us-east-1",
        client=bedrock_client,
    )

    return LLM


def init_embedding_model():
    boto_session = boto3.Session(region_name="us-east-1")
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


def init_retriever():
    boto_session = boto3.Session(region_name="us-east-1")
    sts_client = boto_session.client("sts")

    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::629872170007:role/bedrock-consumer",
        RoleSessionName="be-awesome-dev-bedrock-consumer",
    )

    credentials = assumed_role["Credentials"]

    bedrock_client = boto3.client(
        "bedrock-agent-runtime",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name="us-east-1",
    )

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="KAO6CTXNKO",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
        region_name="us-east-1",
        client=bedrock_client,
    )

    return retriever


def init_compression_retriever():
    llm_for_compressor = init_chat_model()
    compressor = LLMChainExtractor.from_llm(llm_for_compressor)
    retriever = init_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    return compression_retriever


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
