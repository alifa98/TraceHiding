# unlearning_experiments


## Clean Up

```bash
# Remove all checkpoints
find experiments -depth -type d -name '*checkpoints*' -exec rm -rf {} +

```


# GPU clean up
```bash
sudo fuser -k /dev/nvidia5
```
# Monitor stuck processes and killing them

replace the `username` with your username
replace `600` with the limit you want in seconds (10 mins)

```bash 
#!/bin/bash

while true; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -o '[0-9]*' | while read pid; do
        user=$(ps -o user= -p "$pid")
        elapsed=$(ps -o etimes= -p "$pid")
        if [ "$user" = "faraji" ] && [ ! -z "$elapsed" ] && [ "$elapsed" -gt 600 ]; then
            kill -9 "$pid"
            echo "$(date) Killed GPU process $pid (user: faraji, runtime: ${elapsed}s)"
        fi
    done
    sleep 60  # Wait 60 seconds before checking again
done
```

