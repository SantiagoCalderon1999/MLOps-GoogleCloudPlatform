# MLOps MNIST with Vertex AI

This repository contains code for digit classification using the MNIST dataset.

## Technologies Used

The following technologies are utilized in this project:

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Vertex AI](https://cloud.google.com/vertex-ai)
- [GitHub Actions](https://github.com/features/actions)

## Training

The model is trained using the MNIST dataset, which consists of 60,000 images. The primary objective is to accurately classify digits within 28x28 pixel images. Given the relatively clean nature of the dataset, it's advisable to employ data augmentation techniques to diversify the dataset and prevent overfitting.

![MNIST Image](blob/mnist.png)

The model architecture features a Convolutional Neural Network (CNN) comprising three convolutional layers. Batch normalization is applied to mitigate issues like gradient vanishing, and dropout is incorporated to combat overfitting. The network concludes with a softmax layer for classification.

The results of various experiments are documented in the `train-experiment.ipynb` file.

## MLOps Practices

### Requirements

To conduct model training on Google Cloud Platform (GCP), ensure the following prerequisites are met:

1. Enable the Compute Engine API.
2. Enable the Artifact Registry API.
3. Enable the Vertex AI API.

### Pipeline

This repository includes a Continuous Training pipeline in Github Actions that encompasses the following steps:

- **Code Compilation**: Ensure that the code is free from syntax errors and is ready for execution.
- **Model Training with Vertex AI**: Utilize Vertex AI for model training.
- **Model Deployment to Online Prediction Endpoint**: Deploy the trained model to an online prediction endpoint for real-time inference.

To execute and debug the script for training the model, utilize the following command, substituting the variables as specified:

```script
python pipeline/train-model.py --bucket $BUCKET_NAME --container_uri $CONTAINER_URI --vertex_ai_job $VERTEX_AI_JOB --machine_type $MACHINE_TYPE
```

Similarly, to run and debug the script for deploying the model, employ the following command, replacing the variables accordingly:

```script
python pipeline/deploy-model.py --bucket_name $BUCKET_NAME --serving_container_image_uri $CONTAINER_URI --project_id $PROJECT_ID
```

For testing the Docker image created for training purposes, start by building the image with:

```script
docker build -t training_image .
```

Then, run the container using the previously built image:

```script
docker run training_image
```

## Consuming the model

Follow the `consume-model-example.ipynb` notebook to use the deployed endpoint: