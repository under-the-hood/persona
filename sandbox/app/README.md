# Chat with personal CV

## Quick start

Setup environment

```bash
python3 -m venv venv
source venv/bin/activate

pip list |
  cut -d" " -f1 |
  tail -n +3 |
  xargs -n1 pip install --upgrade

pip install -r requirements.txt
```

Run the application

```bash
export OPENAI_API_KEY=...
reflex init
reflex run
```

Finishing

```bash
deactivate
```
