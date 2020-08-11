TODO: make a docs for this

conda and ros do not work very well together...
this env is useful when just want to do stuff with the ml pipeline

clean this up... maybe dont use conda... but have a requirements.txt to install packages

to create conda env go to root directory of this repo
```conda env create -f conda_env.yml --prefix $HOME/anaconda3/envs/autorally python=2.7

to activate env
```conda activate autorally

to update
```https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file
```conda env update -n autorally --file conda_env.yml

to make new yml
```conda env export > conda_env.yml
