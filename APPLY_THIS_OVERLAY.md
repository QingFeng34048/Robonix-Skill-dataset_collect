# Apply this overlay

This archive contains only new or changed Robonix packaging files. Extract it
over the root of `Robonix-Skill-Sim2Real`:

```bash
unzip -o Robonix-Skill-Sim2Real-completed.zip -d Robonix-Skill-Sim2Real
cd Robonix-Skill-Sim2Real
chmod +x scripts/build.sh scripts/start.sh scripts/check_package.py
python3 scripts/check_package.py .
pytest -q
```

Review the changes, fill the real hardware joint limits in the deployment
manifest, then commit them on your own branch.
