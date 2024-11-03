FROM python:3.12.7
COPY ./ /workspaces/src/
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /workspaces/src/
COPY . .