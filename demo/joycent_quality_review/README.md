# Joycent quality review

Run the local review page with the Joycent environment:

```bash
cd /home/xintong/Joycent-code
/home/xintong/miniconda3/envs/joycent/bin/python demo/joycent_quality_review/app.py --port 7861
```

Open `http://<server-host>:7861`. A check mark labels an inference case as high
quality; a cross rejects it. Labels are saved immediately to
`resources/multi_accent/inference/medium_grl/quality_selections.csv`. The page's
download link exports checked cases and their reference paths to
`resources/multi_accent/inference/medium_grl/high_quality_references.csv`.
