from agent import run_agent

with open("job.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

with open("data/resume.txt", "r", encoding="utf-8") as f:
    resume = f.read()

result = run_agent(job_description, resume)

import json
print(json.dumps(result, indent=2))