# Databricks notebook source
import openai
import boto3
import json
import chromadb
import uuid
import os

from pathlib import PurePath
from chromadb import PersistentClient, Settings
from packaging import version
from openai import OpenAI
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType
from botocore.exceptions import NoCredentialsError
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Image
import io

required_version = version.parse("1.10.0")
current_version = version.parse(openai.__version__)

if current_version < required_version:
    raise ValueError(f"Error: OpenAI version {openai.__version__}"
                     " is less than the required version 1.10.0")
else:
    print("OpenAI version is compatible.")

openai.api_key = ""
client = OpenAI(api_key=openai.api_key)
settings = Settings(allow_reset=True)



# Replace the placeholders with your own credentials
ACCESS_KEY = "ENTER IN KEY"
SECRET_KEY = "ENTER IN KEY"

session = boto3.Session(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )

system_prompt = (
   "Please create a detailed narrative description of a person based on the provided characteristics. Here's how you can structure it:"
    "Begin with the full name of the person. Describe their age and, if relevant, how their appearance might have changed with time. Mention their gender identity and race/ethnicity. Provide a detailed description of their physical characteristics, including height, weight" "hair color, and eye color. Detail any distinctive features they have, such as tattoos, scars, or birthmarks. Describe the style of clothing and accessories they prefer, if known."
)

prompt = (
    "Input: A photo of a synthetic picture of a missing person."
    "Instructions: Leverage AI to analyze the provided image and generate a detailed response, just provide the description of the picutre and no suggestive wording."
)


def initiate_chromadb():
    collection_name = "document_embeddings"
    chroma_client = chromadb.PersistentClient(path="./vector_database.db", settings=settings)

    print(chroma_client)
    # Try to get the collection if it exists
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists")
    except Exception as e:
        print(e)
    return collection

def get_metadata_by_id(id, collection):
    """
    Fetch metadata or description for a given ID from the collection.

    Args:
    - id (str): The unique identifier for the document/image.
    - collection: The database collection or table object.

    Returns:
    - str: The description or metadata associated with the given ID.
    """
    # Example query to fetch a document by its ID
    # This is a placeholder and needs to be adapted based on how your database and its API work
    document = collection.find_one({"id": id})

    # Check if the document was found
    if document:
        # Assuming 'description' is the key for the textual information you want
        return document.get('description', 'No description available.')
    else:
        return 'Document not found.'



def add_to_chromadb(collection, embeddings,documents, name):
    # Prepare the document to be inserted
    document = {
        "name": name,
        "embeddings": embeddings
    }
    
    # Attempt to add the document to the collection
    try:
        result = collection.add(
            embeddings = [embeddings],
            ids=[name],
            documents = [documents]
        )
        print(f"Document '{name}' successfully added to the collection.")
        return result
    except Exception as e:
        print(f"Failed to add document '{name}' to the collection: {e}")
        return None


def ask_openai_with_image(prompt, system_prompt, image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",  # Changed model to support JSON mode
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content

def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",  # Use an appropriate model identifier; "gpt-4-1106-preview" might be incorrect or outdated.
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def get_embeddings(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_s3_images():
    s3_client = session.client("s3")
    
    s3_objects = s3_client.list_objects_v2(Bucket="hackathon-data-collection")
    all_images = s3_objects.get('Contents', [])
    
    return all_images

def display_images_from_doc_ids(doc_ids, bucket_name):
    """
    Fetches and displays images from an S3 bucket based on provided document IDs.

    :param doc_ids: A list of document IDs corresponding to image keys in the S3 bucket.
    :param bucket_name: The name of the S3 bucket from which to fetch the images.
    """
    s3_client = session.client("s3")

    for doc_id in doc_ids:
        try:
            # Construct the image key based on the document ID
            # This assumes your document ID directly corresponds to the image key or requires minimal transformation
            image_key = doc_id # Adjust the file extension as necessary

            # Fetch the image from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
            image_data = response['Body'].read()

            # Display the image
            display(Image(data=image_data))
        except Exception as e:
            print(f"Error fetching image for document ID {doc_id}: {e}")

def get_user_input():
    # Ask the user for the type of input they want to provide
    input_type = input("Would you like to enter a text query or upload a picture? Enter 'text' or 'picture': ").strip().lower()

    if input_type == 'text':
        # User opts to enter a text query
        text_query = input("Please enter your text query: ")
        return 'text', text_query
    elif input_type == 'picture':
        # User opts to upload a picture
        # For command-line, ask for the picture file path
        picture_path = input("Please enter the path to your picture file: ")
        return 'picture', picture_path
    else:
        print("Invalid input. Please enter 'text' or 'picture'.")
        return None, None
    
def upload_image_to_s3(file_path, object_name):
    """Upload an image to an S3 bucket"""
    s3_client = session.client("s3")
    try:
        print(object_name)
        bucket_name = "hackathon-data-collection"
        # Use 'file_path' here instead of 'file_name'
        s3_client.upload_file(file_path, bucket_name, object_name)
        
    except NoCredentialsError as e:
        print(f"Credentials not available: {e}")
        return False
    except Exception as e:
        print(e)
        return False
    return True

def get_file_name_from_path(file_path):
    """Extract the file name from the full file path"""
    return PurePath(file_path).name




def get_image_path_by_name(name):
    # Hard-coded mapping of names to their specific paths
    name_to_path_mapping = {
        "david": "/Workspace/Shared/Generative AI Hackathon Event/Image Processing/Images/david.png",
        "eric": "/Workspace/Shared/Generative AI Hackathon Event/Image Processing/Images/eric.png",
        "ryan": "/Workspace/Shared/Generative AI Hackathon Event/Image Processing/Images/ryan.png",
        "marshall": "/Workspace/Shared/Generative AI Hackathon Event/Image Processing/Images/marshall.png",
    }
    
    # Check if the name is in the mapping
    if name in name_to_path_mapping:
        # Return the corresponding pathf
        return name_to_path_mapping[name]
    else:
        # If the name is not found, return None or raise an error
        print(f"Name '{name}' not found in the mapping.")
        return None


def generate_random_object_name(original_filename):
    random_uuid = uuid.uuid4()
    return f"{random_uuid}-{original_filename}"


def generate_presigned_url(object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object"""
    s3_client = session.client("s3")
    try:
        bucket_name="hackathon-data-collection"
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        return None
    return response

def get_document_embeddings(collection):
    result = collection.get(include=["embeddings"])  # Retrieve documents with embeddings
    embeddings = result['embeddings'] if 'embeddings' in result else []
    return embeddings



def retrieve_most_relevant_embeddings(user_query_embedding, document_embeddings, top_n=2):
    # Calculate cosine similarities between user query embedding and all document embeddings
    similarities = cosine_similarity([user_query_embedding], document_embeddings)
    
    # Get indices of top_n most similar chunks
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    
    return top_indices

def retrieve_documents_from_indices(top_indices, collection):
    documents_content = []
    #print(top_indices)
    # Assuming you have a way to map indices to document IDs; adjust accordingly
    document_ids = ['david.png','eric.png', 'george_bjurberg.jpg', 'julian_george_quihuis_1.jpg', 'julian_george_quihuis_2.jpg', 'marshall.png', 'ryan.png']
    for index in top_indices:
        try:
            doc_id = document_ids[index]  # Map index to document ID
            document = collection.get(ids=[doc_id])  # Fetch document using its ID
            #print(document)
            if document:
                documents_content.append(document['documents'])
                
        except Exception as e:
            print(f"Error retrieving document for ID {doc_id}: {str(e)}")
    return documents_content, doc_id

    

def save_text_from_response(response, image_key):
    try:
        # Use the image key as the file name, ensuring it is a valid filename
        # Replace any characters not allowed in filenames with an underscore
        valid_file_name = image_key.replace('/', '_').replace('\\', '_')
        
        # Convert the response to bytes
        file_content = BytesIO(response.encode())

        # S3 Bucket and File Name
        bucket_name = 'missing-persons-data'
        file_name = f"{valid_file_name}.txt"

        # Create an S3 client
        s3_client = session.client("s3")

        # Upload the file
        s3_client.upload_fileobj(file_content, bucket_name, file_name)

        return f"Description saved to S3 as {file_name}"
        
    except Exception as e:
        return str(e)
