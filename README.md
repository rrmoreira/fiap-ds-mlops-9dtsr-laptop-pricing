# Laptop Pricing Predictions Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

*Laptop model pricing prediction*

### Template cookiecutter
```
python -m pipx ensurepath
pipx install cookiecutter-data-science
ccds (na pasta dos projetos)
```


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fiap_ds_mlops_9dtsr_laptop_pricing and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fiap_ds_mlops_9dtsr_laptop_pricing   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fiap_ds_mlops_9dtsr_laptop_pricing a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Dagshub Commands

```
dvc init

dvc add data

git add .gitignore data.dvc

dvc config core.autostage true

git commit -m "Init dvc"

git push origin develop 

dvc remote add origin s3://dvc

dvc remote modify origin endpointurl https://dagshub.com/rrmoreira/fiap-ds-mlops-9dtsr-laptop-pricing.s3

dvc remote modify origin --local access_key_id ****

dvc remote modify origin --local secret_access_key ****

importar o dataset na pasta data/raw 

dvc commit   

dvc remote default origin

dvc push 

```

--------

