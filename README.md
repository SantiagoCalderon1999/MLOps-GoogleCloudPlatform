

Run
```shell
docker build -t training_image .
```

```shell
docker run training_image
```

```shell
python pipeline.train-model.py --bucket <BUCKET_NAME> --project_id <PROJECT_ID>
```