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
npm install && npm run dev
```

By default, the client runs on port `5173`.

### Server

To run the server, create a python virtual environment and install the dependencies in `requirements.txt`. Then, in the root directory, run 

```sh
python3 run.py
```

By default, the server runs on port `3000`. The only function of the server is to parse requests from the client and submit to our model endpoint (see below). 

## Web deployment

The client and flask server is deployed on [Render](https://render.com/). 

## Model deployment

## TODO

- finish the remainder of the models
- update the gallery with actual results