# SenseReact

https://docs.google.com/presentation/d/1PowXbtupLHRTF7QQqd7PE_aklRdRY9yn_GDUYLoicRE/edit?usp=sharing

## How to use satori
To use drag and drop file transfer: access filesystem at https://satori-portal.mit.edu/

Login:
```
ssh [kerb]@satori-login-002.mit.edu
```

Setup conda environment

Running an interactive process:
```
srun --gres=gpu:1 -N 1 --exclusive --mem=250GB --time 8:00:00 --pty /bin/bash
```

Do your work here
```
/nobackup/users/[kerb]
```

## OpenAI
Open a paid openai account at https://beta.openai.com/

Follow the instructions at https://beta.openai.com/docs/api-reference/

## CLIP
To install clip, follow the instructions on their github
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
