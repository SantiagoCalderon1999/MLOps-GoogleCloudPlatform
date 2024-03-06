import argparse
from google.cloud import aiplatform

def run_training_job(bucket, container_uri, job_name, machine_type):
    """
    Run a custom container training job on Google Cloud Vertex AI.

    Args:
        bucket (str): Google Cloud Storage bucket name where training data and artifacts will be stored.
        container_uri (str): URI of the container in Google Cloud Artifact Registry.
        job_name (str): Name of the job in Google Cloud Vertex AI.
        machine_type (str): Machine type for the training job.

    Returns:
        None
    """
    # Create a CustomContainerTrainingJob object
    my_job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=container_uri,
        staging_bucket=f"gs://{bucket}",
    )

    # Run the training job with specified machine type
    my_job.run(
        replica_count=1,
        machine_type=machine_type,
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run custom container training job")
    parser.add_argument("--bucket", required=True, help="Google Cloud Storage bucket name")
    parser.add_argument("--container_uri", required=True, help="Container URI in Google Cloud Artifact Registry")
    parser.add_argument("--vertex_ai_job", required=True, help="Name of the job in Google Cloud Vertex AI")
    parser.add_argument("--machine_type", required=True, help="Machine type for the training job")
    args = parser.parse_args()

    # Call the run_training_job function with parsed arguments
    run_training_job(args.bucket, args.container_uri, args.vertex_ai_job, args.machine_type)
