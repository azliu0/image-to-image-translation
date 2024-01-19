# image-to-image-translation: visualizer and implementation of instructpix2pix

## [https://image.azliu.cc](https://image.azliu.cc)

- Client was built from scratch! Details about this project, including scope, stack, etc. can be found on the [details page](https://www.image.azliu.cc/details). 

- Math details can be found on the [math page](https://www.image.azliu.cc/math). 

- Model details can be found on the [models page](https://www.image.azliu.cc/models).

- Finally, browse some results on the [gallery page](https://www.image.azliu.cc/gallery).

## Setup

### Client

The client is a react app built with yarn: 

```sh
cd client
yarn install && yarn dev
```

### Server

To run the server, create a python virtual environment and install the dependencies in `requirements.txt`. Then, run 

```sh
python3 run.py
```

This is a flask server whose only purpose is to perform inference. All models require around 4GB RAM to run. 

## Note about model deployment

We struggled greatly with figuring out how to deploy our model. Unfortunately, after struggling with many possibilities, we were not able to come up with a good solution for deploying the model in a cost-effective manner. Therefore, the model server is currently running on a spare laptop in Andrew's room. If you have any suggestions for us, please feel free to reach out! For reference, all model inference requires around 4GB RAM, and we would like a solution that we can pay for indefinitely. Serverless is OK.

## TODO

- finish the remainder of the models
- update the gallery with results