![Build Status](https://www.repostatus.org/badges/latest/concept.svg)

# Docker

To deploy both the project and its APIs within a Docker container, you have two Dockerfiles that facilitate deployment and, when required, the exposure of APIs.


[Base Docker File](https://github.com/matheus695p/titanic-dataset/blob/main/data/docker/Dockerfile.base)

[API Docker File](https://github.com/matheus695p/titanic-dataset/blob/main/data/docker/Dockerfile.api)


# Docker docs
This Dockerfile does the following in 9 steps:

1. It starts with a base image using Python 3.10.13.
2. Prints the current working directory inside the container.
3. Copies the `requirements.txt` file from the `src` directory to `/tmp/requirements.txt` in the container.
4. Updates the package list and installs GDAL and other dependencies while cleaning up apt-get cache.
5. Adds a user named `kedro` with specified user and group IDs.
6. Sets the working directory to `/home/kedro`.
7. Grants full permissions to all files and directories under `/home/kedro`.
8. Copies project files from various directories (`conf`, `data`, `docker`, `src`, `notebooks`) into the `/home/kedro` directory in the container.
9. Switches the user to `kedro`, ensuring that the subsequent commands run as the `kedro` user.
10. In case of the API, expose the model serving API in the port 80.
