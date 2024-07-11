## Daint at CSCS

```bash
$ module add daint-gpu
$ module add cray-python/3.8.2.1 

$ python -mvenv env
$ source env/bin/activate
(env) $ pip install --upgrade pip
(env) $ pip install -r requirements.txt -r requirements_optional.txt 
(env) $ pip install -e .
(env) $ module load jupyter-utils
(env) $ pip install ipykernel
(env) $ kernel-create -n d1_9_demo
```
