# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0
RUN python -m pip install jupyter matplotlib numpy pandas scikit-learn seaborn
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-recommended -y
ENV DEBIAN_FRONTEND=dialog

RUN jupyter notebook --generate-config
RUN echo "c=get_config()" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.InlineBackend.rc = { }" >> /root/.jupyter/jupyter_notebook_config.py
WORKDIR /workspace/
RUN apt install -y tree build-essential
RUN ln -sf /usr/bin/python3 /usr/bin/python