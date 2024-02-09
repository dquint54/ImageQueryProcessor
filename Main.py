# Databricks notebook source
!pip install --upgrade openai
!pip install --upgrade chromadb
!pip install --upgrade typing_extensions
dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %run "/Shared/Generative AI Hackathon Event/RAG Application/Utils"

# COMMAND ----------

images = get_s3_images()
collection = initiate_chromadb()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Process incoming file text

# COMMAND ----------

from io import BytesIO

for image in images:
    image_url = generate_presigned_url(image['Key'])
    if image_url:
        name = image['Key']
        response = ask_openai_with_image(prompt=prompt, system_prompt=system_prompt, image_url=image_url)
        print("Response:\n")
        print(response)
        save_status = save_text_from_response(response, name)
        print("\n" + "="*50 + "\n")  # Separator line for readability
        # Processing the response for upload
        try:
            # Check if response is not empty and is a valid JSON string
            if response:
                try:   
                    print("got here")          
                    picture_embeddings = get_embeddings(response)
                    add_to_chromadb(collection=collection, embeddings=picture_embeddings,documents=response,name=name)
                    print(collection.peek())
                except Exception as e:
                    print("Error in processing or uploading the data:", e)                 
            else:
                print("No valid response to process.")
        except Exception as e:
            print(f"Error in processing or uploading the data: {e}")




# COMMAND ----------

# MAGIC %md
# MAGIC ### Enter in user query or submit a picture

# COMMAND ----------

input_type, user_input = get_user_input()

if input_type and user_input:
    if input_type == 'text':
        # Process the text query
        query_embeddings = get_embeddings(user_input)
        picture_embeddings = get_document_embeddings(collection)
        top_indices = retrieve_most_relevant_embeddings(query_embeddings, picture_embeddings)
        documents, doc_ids = retrieve_documents_from_indices(top_indices, collection)
        
        # Ensure doc_ids is a list
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]  # Convert to list if not already
        bucket_name = "hackathon-data-collection"
        display_images_from_doc_ids(doc_ids, bucket_name)


        combined_input = f"Image Description: {user_input}\n\nRelated Information:\n"
        combined_input += f"{documents}\n\n"
        combined_input += "Based on the above, determine if these descriptions match."
        final_response = ask_openai(combined_input)
        print(final_response)

    
    elif input_type == 'picture':
        # Process the picture
        
        file_path = get_image_path_by_name(user_input)
        file_name = get_file_name_from_path(file_path)
        object_name = generate_random_object_name(file_name)
        upload_success = upload_image_to_s3(file_path,object_name=file_name)
        presigned_url = generate_presigned_url(file_name)
        image_text = ask_openai_with_image(system_prompt=system_prompt, prompt=prompt,image_url=presigned_url)
        query_embeddings = get_embeddings(image_text)
        picture_embeddings = get_document_embeddings(collection)
        top_indices = retrieve_most_relevant_embeddings(query_embeddings, picture_embeddings)

        documents, doc_ids = retrieve_documents_from_indices(top_indices, collection)

        # Ensure doc_ids is a list
        if not isinstance(doc_ids, list):
            doc_ids = [doc_ids]  # Convert to list if not already

        bucket_name = "hackathon-data-collection"
        display_images_from_doc_ids(doc_ids, bucket_name)

        combined_input = f"Image Description: {image_text}\n\nRelated Information:\n"
        combined_input += f"{documents}\n\n"
        combined_input += "Based on the above, determine if these descriptions match."
        final_response = ask_openai(combined_input)
        print(final_response)


