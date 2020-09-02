TODO: make a docs for this

need to install during initial setup `sudo apt install python-rosdep`
discuss uneccesary dependencies if just doing simulation stuff
need to add this to zshrc file `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`
install https://www.boost.org/users/history/version_1_73_0.html and look at this issue https://github.com/AutoRally/autorally/issues/88
clean this up...

add stuff about getting stuff to work, i.e. ocs, runstop, joystick
add stuff about path_interal_main node often crashing before working
add stuff explaing predicted vs. actual state controller

to create conda env go to root directory of this repo

add link to sample rosbag
```conda env create -f conda_env.yml --prefix $HOME/anaconda3/envs/autorally python=2.7

to activate env
```conda activate autorally

to update
```https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file
```conda env update -n autorally --file conda_env.yml

to make new yml
```conda env export > conda_env.yml
