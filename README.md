# CAPSTONE 1 PROJECT : ML ZOOMCAMP

## The dataset

In Kaggle, I have found an interesting [dataset](https://www.kaggle.com/datasets/swadeshi/stress-detection-dataset/data) about various psychological, behavioral, and physiological attributes of 100 participants captured daily over 30 days.

The attributes measured are :
- PSS_score: Perceived Stress Scale score, measuring stress levels.
- Openness: Measure of openness to experience, a personality trait.
- Conscientiousness: Measure of conscientiousness, a personality trait.
- Extraversion: Measure of extraversion, a personality trait.
- Agreeableness: Measure of agreeableness, a personality trait.
- Neuroticism: Measure of neuroticism, a personality trait.
- sleep_time: The time (in hours) the participant went to sleep.
- wake_time: The time (in hours) the participant woke up.
- sleep_duration: The duration (in hours) the participant slept.
- PSQI_score: Pittsburgh Sleep Quality Index (PSQI) score, measuring sleep quality.
- call_duration: Total duration of phone calls for the day (in minutes).
- num_calls: Number of phone calls made during the day.
- num_sms: Number of SMS messages sent during the day.
- screen_on_time: Total screen-on time for the day (in hours).
- skin_conductance: Measure of skin conductance, indicating arousal or stress response.
- accelerometer: Accelerometer data representing physical movement.
- mobility_radius: The radius of mobility for the participant (in kilometers).
- mobility_distance: Total distance moved during the day (in kilometers).

## Problem to solve

Stress can be caused by many events and related to specific sets of human attributes.
<br>My goal through this project is to predict the PSS_score (stress level) based on behavioral, psychological, and physiological variables like sleep, mobility, phone usage, and personality traits.
<br>The model should help people control their stress level through an application, once it gets the user's attributes data. The application should afterwards give advices and recommendations to the user.

## Instructions on how to run the project

### Installing the dependecies in a virtual environment
With the Pipfile, you can install the necessary dependencies in a virtual environnement for the project following these steps :
- step 1 : install pipenv with bash command : 
  - pip install pipenv
- step 2 : Navigate to the Project Directory and exactly where the Pipfile is located
- step 3 : Run the following command to install dependencies specified in the Pipfile and create the virtual environment
  <br>*command to execute in bash or terminal* : 
  - pipenv install

### Containerization using Dockerfile
With the DockerFile, you can launch the application in docker container with its dependencies defined in the Pipefile. 
<br>The steps to follow are :
- step 1 : launch Docker Desktop application
- step 2 : Navigate to the Project Directory and exactly where the Dockerfile is located
- step 3 : Run the following command to create the image using the Dockerfile : 
  - docker build -t capstone1_image .
- step 4 : Run the following command to create and run a container using the created docker image : 
  - docker run -it --rm -p 9696:9696 capstone1_image