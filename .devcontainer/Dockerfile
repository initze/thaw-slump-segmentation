# This is the development image, meant to be used within a VSCode devcontainer
# In development, we want to use rye within this devcontainer to manage the package and its dependencies

# With gdal:ubuntu-small-3.9.1 the following versions will be installed:
# - GDAL 3.9.1 <- From the base image
# - Python 3.12.3 <- From the base image
# - git 2.43.0 <- Installed in this image
# - rye (latest version) <- Installed in this image
# - ... on Ubuntu 24.04 LTS <- From the base image
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.1 AS base

USER root
RUN date

# Update distro and install git and rye
RUN apt-get update -y && apt-get upgrade -y \
    && apt-get install git -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.rye/shims:$PATH
RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash
