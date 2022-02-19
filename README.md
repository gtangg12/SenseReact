# SenseReact

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
