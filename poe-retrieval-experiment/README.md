# pool-of-experts



## AUTOML Experiments quickstart

Clone the repo

```
git clone  https://gitlab.fbk.eu/ida/projects/pool-of-experts.git
git checkout automl-experiments
```
Add an ```.env``` file 

```
OPENROUTER_API_KEY=YOUR_API_KEY
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

```
## Current version
The current version of PoE is able to process specific config by calling the function ```AskPoE``` in ```queryPoE.py``` specifing in parameter ```botname``` a path where PoE can find a valid config
