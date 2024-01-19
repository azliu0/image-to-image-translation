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

## TODO

- finish the remainder of the models
- update the gallery with results