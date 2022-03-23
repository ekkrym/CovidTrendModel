FROM renku/renkulab:renku0.10.2-py3.7-0.6.1

# Uncomment and adapt if code is to be included in the image
# COPY src /code/src

# Uncomment and adapt if your R or python packages require extra linux (ubuntu) software
# e.g. the following installs apt-utils and vim; each pkg on its own line, all lines
# except for the last end with backslash '\' to continue the RUN line
#
USER root
# RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    apt-utils \
#    vim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    dirmngr \
    gpg-agent \
    less \
    libcurl4-openssl-dev \
    libxml2-dev \
    libz-dev \
    libgfortran3 \
    software-properties-common
    
USER ${NB_USER}

# install the python dependencies
COPY requirements.txt environment.yml /tmp/
RUN conda env update -q -f /tmp/environment.yml && \
    /opt/conda/bin/pip install -r /tmp/requirements.txt && \
    jupyter labextension install --no-build @jupyter-widgets/jupyterlab-manager &&\
    jupyter labextension install --no-build jupyterlab-plotly@4.11.0 &&\
    jupyter lab build &&\
    conda clean -y --all && \
    conda env export -n "root"