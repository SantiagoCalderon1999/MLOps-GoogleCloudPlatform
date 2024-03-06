from google.cloud import aiplatform

def upload_model(bucket_name, serving_container_image_uri):
    """
    Uploads a model to Google Cloud AI Platform.

    Parameters:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        serving_container_image_uri (str): The URI of the image.

    Returns:
        aiplatform.Model: The uploaded model object.
    """
    # Upload the model
    model = aiplatform.Model.upload(
        display_name='mnist-model',
        artifact_uri=f'gs://{bucket_name}/model_output',
        serving_container_image_uri=serving_container_image_uri
    )
    return model

def deploy_model(project_id, model_id):
    """
    Deploys a model to Google Cloud AI Platform.

    Parameters:
        project_id (str): ID of the project in GCP.
        model_id (str): ID of the model to be deployed.
    """
    # Get the model object
    model = aiplatform.Model(f"projects/{project_id}/locations/us-central1/models/{model_id}") 

    # Deploy the model
    model.deploy(
        deployed_model_display_name='mnist-endpoint',
        traffic_split={"0": 100},
        machine_type="n1-standard-4",
        accelerator_count=0,
        min_replica_count=1,
        max_replica_count=1,
    )

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload a model to Google Cloud AI Platform.")
    parser.add_argument("--bucket_name", type=str, help="The name of the Google Cloud Storage bucket.")
    parser.add_argument("--serving_container_image_uri", type=str, help="The URI of the image.")
    parser.add_argument("--project_id", type=str, help="ID of the project in GCP")
    args = parser.parse_args()

    # Upload the model
    uploaded_model = upload_model(args.bucket_name, args.serving_container_image_uri)

    # Deploy the model
    deploy_model(args.project_id, uploaded_model.name)
