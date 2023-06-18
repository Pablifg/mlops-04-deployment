# mlops-04-deployment
### Goal HM4
Deploy the ride duration model in batch mode.

### Learnings:
- Three ways of deploying a model: web service, streaming, and batch.
- Set up the environment and prepare the workspace.
- Use docker container, build and run.
- Transform of Notebook into a script
- Deploy models with lambda, kasis, prefect in our workflow.


### Result
![](https://pbs.twimg.com/media/Fy3yCxXXsAMvMB2?format=jpg&name=medium)

### Docker commands
```
docker build -t ride-duration-prediction-service:v1 .
```
```
docker run -it --rm ride-duration-prediction-service:v1
```