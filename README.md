Visa-prediction
==============================

# US-Visa-Prediction

## Project overview

This project aims to build a machine learning model to predict the visa application case status (Certified/Denied) based on various factors related to the employee and the employer. The model will utilize employee and employer data, such as education, job experience, wage, job training needs, and more, to make predictions. 


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>




## Dataset Overview 

The dataset contains information about visa applications, including details about the employee, employer, and the visa application case itself. Below is a description of each feature in the dataset

- **case_id**  
  _Description:_ Unique identifier for each visa application.

- **continent**  
  _Description:_ The continent of origin of the employee (e.g., Asia, Europe, etc.).

- **education_of_employee**  
  _Description:_ The level of education of the employee (e.g., Bachelor's, Master's, PhD).

- **has_job_experience**  
  _Description:_ Indicates whether the employee has previous job experience.  
  _Values:_ Y (Yes), N (No).

- **requires_job_training**  
  _Description:_ Indicates if the employee needs job training.  
  _Values:_ Y (Yes), N (No).

- **no_of_employees**  
  _Description:_ The total number of employees in the employer's company.

- **yr_of_estab**  
  _Description:_ The year in which the employer's company was established.

- **region_of_employment**  
  _Description:_ The region of intended employment in the United States (e.g., California, New York, etc.).

- **prevailing_wage**  
  _Description:_ The average wage paid to similarly employed workers in the intended region.

- **unit_of_wage**  
  _Description:_ The unit of the prevailing wage.  
  _Values:_ Hourly, Weekly, Monthly, Yearly.

- **full_time_position**  
  _Description:_ Indicates if the job position is full-time.  
  _Values:_ Y (Full Time), N (Part Time).

- **case_status**  
  _Description:_ The status of the visa application.  
  _Values:_ Certified, Denied.


# Project Workflow Overview

This document outlines the flow and tools used in the project from data management to deployment. The steps include data handling, version control, automation pipelines, experiment tracking, UI development, Dockerization, CI/CD, and finally AWS deployment.

## 1. Data Management

Data management involves storing, organizing, and versioning the data efficiently for smooth collaboration and reproducibility. The data is managed using **DVC (Data Version Control)** to track changes in large datasets and model files over time.

### Key Tasks:
- Store raw data securely.
- Manage data versions using **DVC**.
- Track and retrieve different versions of data and models.
  
## 2. Version Control

**Git** is used for version control to manage the codebase and collaborate with the team. Every change is tracked, and code updates are maintained through branches and commits. Integration with **DVC** ensures data and model files are also versioned along with the code.

### Key Tools:
- **Git** for code version control.
- **GitHub** (or other Git repositories) for remote storage and collaboration.
- **DVC** for tracking large data files.

## 3. Automation with DVC Pipeline

The **DVC pipeline** automates the process of training, evaluation, and model deployment. This pipeline ensures that all the steps in the data processing and model training workflow are repeatable and trackable.

### Key Tasks:
- Define the data processing steps.
- Automate model training and evaluation.
- Reproduce experiments with consistent results using the DVC pipeline.

## 4. Experiment Tracking with MLflow

**MLflow** is used for experiment tracking, allowing us to log parameters, metrics, and outputs of our machine learning models. This helps in comparing different models, tracking experiments, and improving model performance.

### Key Tasks:
- Track machine learning experiments.
- Log hyperparameters, metrics, and artifacts.
- Organize and compare results from different experiments.

## 5. Streamlit UI

A user-friendly web application is built using **Streamlit** for model deployment and visualization. Streamlit allows easy creation of dashboards and interactive UIs to showcase the model's predictions, results, and insights.

### Key Tasks:
- Build interactive UI for model predictions and visualizations.
- Allow users to upload new data and get real-time results.
- Present model evaluation metrics and insights in a clean, user-friendly manner.

## 6. Dockerization

To ensure the project runs consistently across different environments, we use **Docker** to containerize the application. Docker allows packaging the code, dependencies, and configurations into a single container that can be run anywhere.

### Key Tasks:
- Create a `Dockerfile` to containerize the application.
- Ensure dependencies and environment configurations are consistent.
- Build and run the application as a Docker container.

## 7. CI/CD (Continuous Integration/Continuous Deployment)

We use **CI/CD pipelines** to automate testing, building, and deployment processes. This ensures that any new changes are quickly validated and deployed without manual intervention, leading to faster and more reliable releases.

### Key Tools:
- **GitHub Actions**, **Jenkins**, or **GitLab CI** for CI/CD pipelines.
- Automated tests for code and model performance.
- Continuous deployment to ensure automatic updates on changes.

### Key Tasks:
- Automate testing and building of the application.
- Deploy the application automatically to development or production environments.
- Ensure high-quality code through automated linting, testing, and validation.

## 8. AWS Deployment

Finally, the application is deployed to **AWS** (Amazon Web Services). This ensures scalability, reliability, and security for hosting the application in a cloud environment.

### Key Tasks:
- Deploy the Dockerized application to **AWS** (e.g., EC2, EKS).
- Set up **AWS S3** for storage of data and model files.
- Use **AWS Lambda** or **AWS API Gateway** for scalable serverless functions.

### Key AWS Services:
- **AWS EC2** for running virtual machines.
- **AWS S3** for storage.
- **AWS ECR** for managing containers.


## Summary

This project follows a structured pipeline from data management and version control to automation, experiment tracking, UI development, Dockerization, CI/CD, and deployment on AWS. By using these tools and practices, we ensure the project is efficient, scalable, and reproducible.


# workflow 


## installing cookicutter


```bash
pip install cookiecutter 
```
```bash
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```


## creating virtual environment

```bash
conda create -p visa python==3.10 -y
```

## activating environment


```bash
conda activate visa/
```

## installing requiremnts

```bash
pip install requiremnts.txt 
```